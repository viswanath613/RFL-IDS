import pandas as pd
import glob
import os

# ==============================
# CONFIG PATHS
# ==============================
CICIDS_PATH = "datasets/CICIDS2017/*.csv"
UNSW_TRAIN = "datasets/UNSW_NB15_training-set.csv"
UNSW_TEST = "datasets/UNSW_NB15_testing-set.csv"
TON_PATH = "datasets/TON_IoT/Train_Test_Network.csv"

OUTPUT_PATH = "data/raw/dataset.csv"

# ==============================
# CREATE DATASET
# ==============================
def create_dataset():
    dfs = []

    # ---- CICIDS ----
    cic_files = glob.glob(CICIDS_PATH)
    for f in cic_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except:
            print(f"Skipped: {f}")

    print(f"CICIDS loaded: {len(dfs)} files")

    # ---- UNSW ----
    try:
        unsw_train = pd.read_csv(UNSW_TRAIN)
        unsw_test = pd.read_csv(UNSW_TEST)
        dfs.append(unsw_train)
        dfs.append(unsw_test)
        print("UNSW loaded")
    except:
        print("UNSW not found")

    # ---- TON_IoT ----
    try:
        ton = pd.read_csv(TON_PATH)
        dfs.append(ton)
        print("TON_IoT loaded")
    except:
        print("TON_IoT not found")

    # ---- Merge all ----
    final_df = pd.concat(dfs, ignore_index=True)

    print("Total rows:", len(final_df))

    # ==============================
    # CLEANING
    # ==============================
    final_df = final_df.dropna()

    # Convert label column (last column)
    if final_df.iloc[:, -1].dtype == 'object':
        final_df.iloc[:, -1] = final_df.iloc[:, -1].astype('category').cat.codes

    # ==============================
    # SAVE FILE
    # ==============================
    os.makedirs("data/raw", exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved at: {OUTPUT_PATH}")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    create_dataset()
