import os
import random
import json
from pycocotools.coco import COCO

# Load COCO Annotations
train_coco = COCO('/workspace/CLDETR/data/coco/annotations/instances_train2017.json')
val_coco = COCO('/workspace/CLDETR/data/coco/annotations/instances_val2017.json')

# Parameters for subset creation
num_images_per_class = 40

# Helper function to select images for each category
def get_subset(coco, num_images_per_class):
    selected_image_ids = set()
    for cat_id in coco.getCatIds():
        img_ids = coco.getImgIds(catIds=[cat_id])
        selected_img_ids = random.sample(img_ids, min(num_images_per_class, len(img_ids)))
        selected_image_ids.update(selected_img_ids)
    
    # Collect images and annotations for subset
    subset_annotations = {
        "images": [img for img in coco.dataset['images'] if img['id'] in selected_image_ids],
        "annotations": [ann for ann in coco.dataset['annotations'] if ann['image_id'] in selected_image_ids],
        "categories": coco.dataset['categories']
    }
    return subset_annotations

# Generate train and val subsets
train_subset = get_subset(train_coco, num_images_per_class)
val_subset = get_subset(val_coco, num_images_per_class)

# Save subset annotations
os.makedirs('/workspace/coco40/annotations', exist_ok=True)
with open('/workspace/coco40/annotations/instances_train2017.json', 'w') as f:
    json.dump(train_subset, f)
with open('/workspace/coco40/annotations/instances_val2017.json', 'w') as f:
    json.dump(val_subset, f)
