import json
import os
import shutil

# Paths
original_train_dir = '/workspace/CLDETR/data/coco/train2017'  # Path to the original COCO train images
original_val_dir = '/workspace/CLDETR/data/coco/val2017'      # Path to the original COCO val images
subset_train_dir = '/workspace/coco40/train2017'    # Path where subset train images will be saved
subset_val_dir = '/workspace/coco40/val2017'        # Path where subset val images will be saved

# Ensure directories exist
os.makedirs(subset_train_dir, exist_ok=True)
os.makedirs(subset_val_dir, exist_ok=True)

# Load subset annotations
with open('/workspace/coco40/annotations/instances_train2017.json', 'r') as f:
    train_annotations = json.load(f)

with open('/workspace/coco40/annotations/instances_val2017.json', 'r') as f:
    val_annotations = json.load(f)

# Helper function to copy images based on subset annotations
def copy_images(subset_annotations, original_dir, subset_dir):
    for image_info in subset_annotations['images']:
        file_name = image_info['file_name']
        original_path = os.path.join(original_dir, file_name)
        subset_path = os.path.join(subset_dir, file_name)
        
        # Copy image if it exists in the original directory
        if os.path.exists(original_path):
            shutil.copy2(original_path, subset_path)
        else:
            print(f"Warning: {file_name} not found in {original_dir}")

# Copy train and validation images
copy_images(train_annotations, original_train_dir, subset_train_dir)
copy_images(val_annotations, original_val_dir, subset_val_dir)

print("Subset images have been copied to the designated directories.")
