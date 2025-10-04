import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ===== ADJUST THESE PATHS =====
base_folder = "data"
csv_file = "data/Chest_xray_Corona_Metadata.csv"
# ==============================

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv(csv_file)
print(f"CSV loaded successfully! Total rows: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Check what's in the Dataset_type column
print(f"\nDataset types available: {df['Dataset_type'].unique()}")
print(f"\nLabels available: {df['Label'].unique()}")

# Filter for TRAIN dataset (change to TEST if needed)
df_filtered = df[df['Dataset_type'] == 'TRAIN']
print(f"\nFiltered dataset size: {len(df_filtered)}")

# Get samples from each class
normal_samples = df_filtered[df_filtered['Label'] == 'Normal'].head(3)
pneumonia_samples = df_filtered[df_filtered['Label'] == 'Pnemonia'].head(2)

# If 'Pnemonia' doesn't exist, try 'Pneumonia'
if len(pneumonia_samples) == 0:
    pneumonia_samples = df_filtered[df_filtered['Label'] == 'Pneumonia'].head(2)

samples = pd.concat([normal_samples, pneumonia_samples])

print(f"\nSearching for images...")
print(f"Sample image names from CSV:")
for img_name in samples['X_ray_image_name'].head():
    print(f"  - {img_name}")

# Try different possible folder structures
possible_paths = [
    base_folder,
    os.path.join(base_folder, "train"),
    os.path.join(base_folder, "test"),
    os.path.join(base_folder, "data", "train"),
    os.path.join(base_folder, "data", "test"),
]

# Find where images actually are
found_path = None
for path in possible_paths:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            print(f"\nFound {len(files)} images in: {path}")
            print(f"Sample files: {files[:3]}")
            found_path = path
            break

if found_path is None:
    print("\n❌ ERROR: Could not find image folder!")
    print("Please check your folder structure and update the paths.")
else:
    print(f"\n✓ Using image folder: {found_path}")

    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Sample Chest X-Ray Images: Normal vs Pneumonia', fontsize=16, fontweight='bold')

    # Display images
    for idx, (i, row) in enumerate(samples.iterrows()):
        if idx >= 5:
            break

        img_name = row['X_ray_image_name']
        label = row['Label']

        img_path = os.path.join(found_path, img_name)

        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(img, cmap='gray')
            color = 'red' if 'nemonia' in label else 'black'
            axes[idx].set_title(f'{label}\n{img_name[:15]}...', fontsize=10, color=color)
            print(f"✓ Loaded: {img_name}")
        else:
            axes[idx].text(0.5, 0.5, f'Not found:\n{img_name}',
                           ha='center', va='center', fontsize=8, wrap=True)
            axes[idx].set_title(f'{label}', fontsize=10)
            print(f"✗ Not found: {img_path}")

        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

# Print statistics
print(f"\n{'=' * 50}")
print(f"Dataset Statistics:")
print(f"{'=' * 50}")
print(df_filtered['Label'].value_counts())
print(f"\nTotal images in filtered dataset: {len(df_filtered)}")
print(f"{'=' * 50}")

print(df)
print()