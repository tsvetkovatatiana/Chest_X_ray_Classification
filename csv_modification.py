#%%
# Imports
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

#%%
metadata = pd.read_csv("data/Chest_xray_Corona_Metadata.csv")
cleaned_csv = metadata.copy()

# fix typo "Pnemonia" to "pneumonia"
cleaned_csv['Label'] = cleaned_csv['Label'].str.strip().str.lower()
cleaned_csv['Label'] = cleaned_csv['Label'].replace({'pnemonia': 'pneumonia'})

#%%
def build_path(row: pd.Series) -> str:
    dataset_dir = os.path.join("data", row["Dataset_type"].lower())
    path_to_img = os.path.join(dataset_dir, row["X_ray_image_name"])
    return path_to_img

cleaned_csv["Path"] = cleaned_csv.apply(build_path, axis=1)

cleaned_csv = cleaned_csv.rename(columns={'X_ray_image_name': 'Name'})

#%%
cleaned_csv = cleaned_csv[['Name', 'Path', 'Label', 'Dataset_type']]

#%%
os.makedirs("data/CSVs", exist_ok=True)
cleaned_csv.to_csv("data/CSVs/cleaned_full_metadata.csv", index=False)
print("Cleaned metadata saved to data/CSVs/cleaned_full_metadata.csv")

#%%
train_df = cleaned_csv[cleaned_csv["Dataset_type"].str.upper() == "TRAIN"].copy()
# 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df["Label"]), 1):
    train_fold = train_df.iloc[train_index]
    val_fold = train_df.iloc[val_index]

    # Save to CSV files
    train_fold.to_csv(f"data/CSVs/fold_{fold}_train.csv", index=False)
    val_fold.to_csv(f"data/CSVs/fold_{fold}_val.csv", index=False)

print("Created 5-fold CSVs in data/CSVs/")


#%%
# Now create a cleaned csv for the test data
test_df = cleaned_csv[cleaned_csv["Dataset_type"].str.upper() == "TEST"].copy()
test_df.to_csv(f"data/CSVs/test_data.csv", index=False)
