import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

# =========================================
# 1. CICIDS-2017 (HuggingFace)
# =========================================
print("Loading CICIDS...")

cic = load_dataset("veera33/CIC-IDS2017")

df_cic = cic['train'].to_pandas()
df_cic = df_cic.dropna()

# Ensure label is numeric
if df_cic.iloc[:, -1].dtype == 'object':
    df_cic.iloc[:, -1] = df_cic.iloc[:, -1].astype('category').cat.codes

df_cic.to_csv("data/raw/cicids.csv", index=False)
print("Saved: data/raw/cicids.csv")

# =========================================
# 2. UNSW-NB15 (Manual Download Required)
# =========================================
print("Processing UNSW...")

try:
    train = pd.read_csv("datasets/UNSW_NB15_training-set.csv")
    test = pd.read_csv("datasets/UNSW_NB15_testing-set.csv")

    df_unsw = pd.concat([train, test], ignore_index=True)
    df_unsw = df_unsw.dropna()

    if df_unsw.iloc[:, -1].dtype == 'object':
        df_unsw.iloc[:, -1] = df_unsw.iloc[:, -1].astype('category').cat.codes

    df_unsw.to_csv("data/raw/unsw_nb15.csv", index=False)
    print("Saved: data/raw/unsw_nb15.csv")

except:
    print("UNSW dataset not found. Please download manually.")

# =========================================
# 3. TON_IoT (Manual Download Required)
# =========================================
print("Processing TON_IoT...")

try:
    df_ton = pd.read_csv("datasets/TON_IoT/Train_Test_Network.csv")
    df_ton = df_ton.dropna()

    if df_ton.iloc[:, -1].dtype == 'object':
        df_ton.iloc[:, -1] = df_ton.iloc[:, -1].astype('category').cat.codes

    df_ton.to_csv("data/raw/ton_iot.csv", index=False)
    print("Saved: data/raw/ton_iot.csv")

except:
    print("TON_IoT dataset not found. Please download manually.")

print("All datasets processed!")
