# tunnel dimensions: width = 5m, height = 5.5m, length = 8200m
# surface height limits:
#   limit 1: 0.4m - 350m3
#   limit 2: 5.9m - 75975m3
#   limit 3: 8.6m - 150225m3
#   limit 4: 14.1m - 225850m3

import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier

# Read the Excel file (first two columns: level + volume)
convtable_raw = pd.read_excel(
    "./raw_data/Volume of tunnel vs level Blominmäki.xlsx",  # adjust path if needed
    sheet_name="Taul1"
)
# Drop the "Formula type" column (last one)
convtable_raw = convtable_raw.iloc[:, :2]  # keep only Level + Volume
# Optional: rename for convenience (not required)
convtable_raw.columns = ["level", "volume"]
# Make sure it's sorted by level
convtable_raw = convtable_raw.sort_values("level").reset_index(drop=True)

def get_vol(level):
    levels = convtable_raw["level"].to_numpy()
    vols = convtable_raw["volume"].to_numpy()

    # Handle outside range: clamp to min/max volume
    if level <= levels[0]:
        return vols[0]
    if level >= levels[-1]:
        return vols[-1]

    # Find position where this level would be inserted to keep order
    i = np.searchsorted(levels, level)

    # Interpolate between (i-1) and i
    x0, x1 = levels[i-1], levels[i]
    y0, y1 = vols[i-1], vols[i]

    return y0 + (level - x0) * (y1 - y0) / (x1 - x0)



raw = pd.read_excel("./raw_data/Hackathon_HSY_data.xlsx")

print(raw.shape)
print(raw.columns)
print(raw.dtypes)
print(raw.head())
print(raw.describe(include='all').T)


df = raw.copy()

# Rename if necessary
# df = df.rename(columns={'Time stampColumnName': 'Time stamp'})

df['Time stamp'] = pd.to_datetime(df['Time stamp'], errors='coerce')
df = df.dropna(subset=['Time stamp'])    # drop rows with broken Time stamp parsing
df = df.sort_values('Time stamp')
df = df.set_index('Time stamp')

# e.g. 15-minute data; choose rule to suit the dataset
df = df.resample('15T').mean()

constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
df = df.drop(columns=constant_cols)


df = df.ffill().bfill()

numeric_cols = df.select_dtypes(include='number').columns

# Clip each numeric column to [1st, 99th] percentile to kill spikes
q_low = df[numeric_cols].quantile(0.02)
q_hi  = df[numeric_cols].quantile(0.99)

df[numeric_cols] = df[numeric_cols].clip(q_low, q_hi, axis=1)

df.to_csv("./output/cleaned_data.csv", index=True)



