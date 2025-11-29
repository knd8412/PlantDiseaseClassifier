import os
import zipfile
import kagglehub
import shutil
import sys
import warnings
from clearml import Dataset

# Add data folder to path to use our utility functions
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from data.utils import build_class_mapping, gather_samples, make_subset

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
PROJECT_NAME = "Datasets/PlantVillage"
KAGGLE_HANDLE = "abdallahalidev/plantvillage-dataset"
RANDOM_SEED = 42  # For reproducibility
MODALITIES = ["color", "grayscale", "segmented"]  # All modalities to include

# Define dataset configurations: (name, ratio, description)
DATASET_CONFIGS = [
    ("PlantVillage-Tiny", 0.05, "5% stratified subset for quick prototyping"),
    ("PlantVillage-Medium", 0.30, "30% stratified subset for hyperparameter tuning"),
    ("PlantVillage-Large", 0.60, "60% stratified subset for full training"),
]

# ---------------------------------------------------
# -1. Delete specific broken datasets (User Request)
# ---------------------------------------------------
def delete_dataset(dataset_id):
    try:
        print(f"Attempting to delete dataset {dataset_id}...")
        Dataset.delete(dataset_id)
        print(f"Successfully deleted dataset {dataset_id}")
    except Exception as e:
        print(f"Error deleting dataset {dataset_id}: {e}")

# IDs provided by user to be deleted
DATASETS_TO_DELETE = [
    "ff19664e8eea4e79a93d69f6dae54716",  # Medium
    "a20b80fd8e85450d9db29dc867a13c3e"   # Large
]

print(f"Deleting {len(DATASETS_TO_DELETE)} specific datasets before processing...")
for ds_id in DATASETS_TO_DELETE:
    delete_dataset(ds_id)

# ---------------------------------------------------
# 0. Check which datasets already exist in ClearML
# ---------------------------------------------------
print("Checking which datasets already exist in ClearML...")
datasets_to_create = []

for dataset_name, ratio, description in DATASET_CONFIGS:
    try:
        existing_ds = Dataset.get(dataset_name=dataset_name, dataset_project=PROJECT_NAME)
        print(f"  ✓ {dataset_name} already exists (ID: {existing_ds.id})")
    except:
        print(f"  ✗ {dataset_name} not found, will create")
        datasets_to_create.append((dataset_name, ratio, description))

if not datasets_to_create:
    print("\nAll datasets already uploaded to ClearML!")
    exit(0)

print(f"\nWill create {len(datasets_to_create)} dataset(s)...\n")

# ---------------------------------------------------
# 1. Check if Kaggle dataset already downloaded locally
# ---------------------------------------------------
print("Checking if KaggleHub has dataset cached...")

cache_dir = os.path.expanduser("~/.cache/kagglehub/datasets")
dataset_cache_root = os.path.join(cache_dir, KAGGLE_HANDLE.replace("/", os.sep))

if os.path.exists(dataset_cache_root):
    print("Local Kaggle cache found:", dataset_cache_root)
    # Find the newest version folder
    versions = [
        os.path.join(dataset_cache_root, v)
        for v in os.listdir(dataset_cache_root)
        if os.path.isdir(os.path.join(dataset_cache_root, v))
    ]

    if len(versions) > 0:
        versions_path = max(versions, key=os.path.getmtime)
        
        # The actual data is in nested folders, find the one with modalities
        def find_dataset_root(path, depth=0, max_depth=3):
            """Recursively find folder containing color/grayscale/segmented folders"""
            if depth > max_depth:
                return None
            
            # Check if this folder has the modality folders
            try:
                contents = os.listdir(path)
                if all(mod in contents for mod in ["color", "grayscale", "segmented"]):
                    return path
            except:
                return None
            
            # Recursively search subfolders
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    result = find_dataset_root(item_path, depth + 1, max_depth)
                    if result:
                        return result
            return None
        
        local_path = find_dataset_root(versions_path)
        if not local_path:
            local_path = versions_path  # Fallback
        print("Using cached Kaggle dataset:", local_path)
    else:
        print("Cache folder exists but no versions found, downloading...")
        local_path = kagglehub.dataset_download(KAGGLE_HANDLE)

else:
    print("No local cache, downloading dataset from Kaggle...")
    local_path = kagglehub.dataset_download(KAGGLE_HANDLE)

print("Dataset path:", local_path)

# ---------------------------------------------------
# 2. Use data.utils functions to gather all samples
# ---------------------------------------------------
print("Building class mapping and gathering samples...")
class_names, class_to_idx = build_class_mapping(local_path, modality="color")
all_samples = gather_samples(local_path, MODALITIES, class_to_idx)

print(f"  Total samples found: {len(all_samples)}")
print(f"  Number of classes: {len(class_names)}")
print(f"  Modalities: {MODALITIES}")


# ---------------------------------------------------
# 3. Create and upload each dataset using data.utils.make_subset
# ---------------------------------------------------
for dataset_name, subset_ratio, description in datasets_to_create:
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name} ({subset_ratio*100:.0f}%)")
    print(f"{'='*60}")
    
    # Create stratified subset using data.utils.make_subset
    print(f"Creating stratified subset using make_subset({subset_ratio*100:.0f}%)...")
    subset_samples = make_subset(all_samples, ratio=subset_ratio, seed=RANDOM_SEED)
    
    print(f"  Total samples available: {len(all_samples)}")
    print(f"  Sampled samples (stratified): {len(subset_samples)}")
    print(f"  Reduction: {(1 - len(subset_samples)/len(all_samples))*100:.1f}%")
    
    # -------------------------------------------------------
    # CHUNKING LOGIC
    # Split dataset into multiple zips (e.g., 250MB chunks)
    # to prevent upload failures from restarting the whole process.
    # -------------------------------------------------------
    CHUNK_SIZE_MB = 250
    MAX_CHUNK_BYTES = CHUNK_SIZE_MB * 1024 * 1024
    
    base_filename = dataset_name.lower().replace('-', '_')
    
    # Clean up any existing parts from previous runs
    for f in os.listdir(local_path):
        if f.startswith(base_filename) and f.endswith(".zip"):
            try:
                os.remove(os.path.join(local_path, f))
            except:
                pass

    print(f"Zipping {dataset_name} into chunks of ~{CHUNK_SIZE_MB}MB...")
    
    created_zips = []
    current_part = 1
    current_zip_name = f"{base_filename}_part_{current_part:03d}.zip"
    current_zip_path = os.path.join(local_path, current_zip_name)
    
    current_chunk_size = 0
    file_count = 0
    
    # Open first zip
    zipf = zipfile.ZipFile(current_zip_path, "w", zipfile.ZIP_DEFLATED)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for img_path, label_id, modality in subset_samples:
            # img_path is absolute, make it relative to local_path for archive
            arcname = os.path.relpath(img_path, local_path)
            
            # Write file
            zipf.write(img_path, arcname)
            
            # Track size (using uncompressed size as proxy for chunking logic)
            current_chunk_size += os.path.getsize(img_path)
            file_count += 1
            
            # If chunk is full, close and start next
            if current_chunk_size >= MAX_CHUNK_BYTES:
                zipf.close()
                created_zips.append(current_zip_path)
                print(f"  ✓ Created part {current_part}: {current_zip_name} ({current_chunk_size/1024/1024:.1f} MB uncompressed)")
                
                current_part += 1
                current_chunk_size = 0
                current_zip_name = f"{base_filename}_part_{current_part:03d}.zip"
                current_zip_path = os.path.join(local_path, current_zip_name)
                zipf = zipfile.ZipFile(current_zip_path, "w", zipfile.ZIP_DEFLATED)

        # Close the final zip
        zipf.close()
        
        # Add final zip if it has content
        if os.path.exists(current_zip_path) and os.path.getsize(current_zip_path) > 0:
            created_zips.append(current_zip_path)
            print(f"  ✓ Created part {current_part}: {current_zip_name}")
        else:
            if os.path.exists(current_zip_path):
                os.remove(current_zip_path)

    print(f"Zipping complete. Total files: {file_count}. Total parts: {len(created_zips)}")
    
    # Upload to ClearML
    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=PROJECT_NAME,
        description=f"{description} - Downloaded via KaggleHub - Stratified - Chunked Upload"
    )
    
    # Add all zip parts one by one with a delay to avoid simultaneous uploads
    import time  # Import time for adding delays

    for zip_part in created_zips:
        print(f"Adding file {zip_part} to dataset...")
        dataset.add_files(zip_part)
        dataset.upload()
    
    dataset.finalize()
    
    print(f"✅ {dataset_name} uploaded! Dataset ID: {dataset.id}")
    
    # Cleanup zips after upload
    print("Cleaning up local zip parts...")
    for zip_part in created_zips:
        if os.path.exists(zip_part):
            os.remove(zip_part)

# ---------------------------------------------------
# 4. Final cleanup: Delete Kaggle dataset
# ---------------------------------------------------
print(f"\n{'='*60}")
print("Final cleanup...")
print(f"{'='*60}")

if os.path.exists(local_path):
    shutil.rmtree(local_path)
    print("Deleted Kaggle dataset:", local_path)

print("\n✅ All datasets uploaded successfully!")
print("Cleanup complete!")
