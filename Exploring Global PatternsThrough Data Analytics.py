#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1 — Basic setup & imports
# Run this first. If you use a fresh environment, install missing packages with pip.
import sys
import os
print('Python', sys.version)


# Standard packages we will use
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# Helpful display settings for Jupyter
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)


# Create output folder
OUT_DIR = 'debt_outputs'
os.makedirs(OUT_DIR, exist_ok=True)
print('Output folder:', OUT_DIR)


# %% [markdown]
# Why we chose these World Bank indicators
# - `GC.DOD.TOTL.GD.ZS` = Central government debt, total (% of GDP)
# - `DT.DOD.DPPG.CD` = External debt stocks, public and publicly guaranteed (current US$)
#
# We'll fetch both, then pick the most recent year available per country for comparison.


# In[3]:


# Cell 2 — Helper function: fetch World Bank indicator
# This returns a DataFrame with columns: country_code, country, year, value

def fetch_wb_indicator(indicator_code, per_page=20000):
    """
    Fetch World Bank indicator for all countries (large per_page to reduce pagination).
    Returns pandas DataFrame: country_code, country, year (int), value (float or NaN).
    """
    base = f"http://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    url = f"{base}?format=json&per_page={per_page}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank response; got: {payload}")
    df = pd.json_normalize(payload[1])
    # Keep only needed columns
    df = df[['country.id', 'country.value', 'date', 'value']]
    df.columns = ['country_code', 'country', 'year', 'value']
    # Convert types
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

print('Helper defined: fetch_wb_indicator')


# In[4]:


# Cell 3 — Fetch the two indicators (run this cell to download)
# NOTE: This contacts the World Bank API. Ensure you have an internet connection.

print("Fetching central government debt (% of GDP)...")
debt_pct = fetch_wb_indicator("GC.DOD.TOTL.GD.ZS")
print("Rows fetched for debt_pct:", len(debt_pct))

print("Fetching external public debt (PPG) in USD...")
external_ppg = fetch_wb_indicator("DT.DOD.DPPG.CD")
print("Rows fetched for external_ppg:", len(external_ppg))

# Quick peek at first few rows
print("\n--- debt_pct sample ---")
print(debt_pct.head())

print("\n--- external_ppg sample ---")
print(external_ppg.head())


# In[5]:


# Cell 4 — Reduce to most recent non-null value per country

def latest_by_country(df, value_col='value'):
    df_nonnull = df[df[value_col].notnull()].copy()
    # Ensure year is numeric and drop missing years
    df_nonnull = df_nonnull[df_nonnull['year'].notnull()]
    df_nonnull['year'] = df_nonnull['year'].astype(int)
    
    # Sort and take the latest year for each country
    idx = (
        df_nonnull.sort_values(['country', 'year'])
        .groupby('country')
        .tail(1)
        .set_index('country')
    )
    
    return idx[['country_code', 'year', value_col]].rename(columns={value_col: value_col})

print("Computing latest non-missing values per country...")

debt_pct_latest = latest_by_country(debt_pct, 'value').rename(columns={'value': 'debt_pct_gdp'})
external_ppg_latest = latest_by_country(external_ppg, 'value').rename(columns={'value': 'external_ppg_usd'})

print("Countries with debt_pct_latest:", len(debt_pct_latest))
print("Countries with external_ppg_latest:", len(external_ppg_latest))


# In[6]:


# Cell 5 — Top 15: debt (% GDP) and Top 15: external public debt (USD)
TOP_N = 15

# Top countries by debt as % of GDP
top_debt_pct = debt_pct_latest.sort_values('debt_pct_gdp', ascending=False).head(TOP_N)
top_debt_pct.to_csv(os.path.join(OUT_DIR, 'top_debt_pct_gdp.csv'))

# Top countries by external public debt (USD)
top_external_usd = external_ppg_latest.sort_values('external_ppg_usd', ascending=False).head(TOP_N)
top_external_usd.to_csv(os.path.join(OUT_DIR, 'top_external_debt_usd.csv'))

print("\nTop 15 by debt (% of GDP):")
print(top_debt_pct)

print("\nTop 15 by external public debt (USD):")
print(top_external_usd)


# In[7]:


# Cell 6 — Visualize Top 15 Indebted Nations with Bar Charts
import matplotlib.pyplot as plt

# Debt as % of GDP ---
plt.figure(figsize=(10, 6))
plt.barh(top_debt_pct.index[::-1], top_debt_pct["debt_pct_gdp"][::-1])
plt.title("Top 15 Most Indebted Countries (Debt % of GDP)", fontsize=14, weight='bold')
plt.xlabel("Debt (% of GDP)")
plt.ylabel("Country")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# External Public Debt (USD) ---
plt.figure(figsize=(10, 6))
plt.barh(top_external_usd.index[::-1], top_external_usd["external_ppg_usd"][::-1] / 1e9)
plt.title("Top 15 Countries by External Public Debt (USD, Billions)", fontsize=14, weight='bold')
plt.xlabel("Debt (USD Billions)")
plt.ylabel("Country")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[8]:


# Cell 7 — Interactive dashboard for debt analysis
import plotly.express as px

# Interactive: Debt as % of GDP ---
fig1 = px.bar(
    top_debt_pct,
    x="debt_pct_gdp",
    y=top_debt_pct.index,
    orientation='h',
    color="debt_pct_gdp",
    color_continuous_scale="Reds",
    title="Top 15 Most Indebted Countries (Debt % of GDP)",
    labels={"debt_pct_gdp": "Debt (% of GDP)", "country": "Country"},
)
fig1.update_layout(xaxis_title="Debt (% of GDP)", yaxis_title="Country", template="plotly_white")
fig1.show()

# Interactive: External Public Debt (USD) ---
fig2 = px.bar(
    top_external_usd,
    x=top_external_usd["external_ppg_usd"] / 1e9,
    y=top_external_usd.index,
    orientation='h',
    color=top_external_usd["external_ppg_usd"],
    color_continuous_scale="Blues",
    title="Top 15 Countries by External Public Debt (USD Billions)",
    labels={"x": "Debt (USD Billions)", "country": "Country"},
)
fig2.update_layout(xaxis_title="Debt (USD Billions)", yaxis_title="Country", template="plotly_white")
fig2.show()


# In[15]:


# Save Matplotlib figures
plt.savefig("top_external_debt_usd.png", dpi=300, bbox_inches="tight")


# In[ ]:




