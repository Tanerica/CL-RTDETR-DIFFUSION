import requests
import os
import random
import json
from pycocotools.coco import COCO

# Paths
detection_annotations_path = 'path/to/annotations/instances_train2017.json'
captions_annotations_path = 'path/to/annotations/captions_train2017.json'
output_image_dir = 'path/to/subset_images/train2017'
output_detection_ann_path = 'path/to/subset_annotations/instances_train2017_subset.json'
output_captions_ann_path = 'path/to/subset_annotations/captions_train2017_subset.json'
os.makedirs(output_image_dir, exist_ok=True)

# Load COCO Annotations
detection_coco = COCO(detection_annotations_path)
captions_coco = COCO(captions_annotations_path)
num_images_per_class = 100

# Get a list of selected image IDs
selected_image_ids = set()
for cat_id in detection_coco.getCatIds():
    img_ids = detection_coco.getImgIds(catIds=[cat_id])
    selected_img_ids = random.sample(img_ids, min(num_images_per_class, len(img_ids)))
    selected_image_ids.update(selected_img_ids)

# Download each selected image by constructing the URL
base_url = "http://images.cocodataset.org/train2017/"
for img_id in selected_image_ids:
    img_info = detection_coco.loadImgs(img_id)[0]
    img_url = f"{base_url}{img_info['file_name']}"
    img_path = os.path.join(output_image_dir, img_info['file_name'])
    
    # Download image if not already downloaded
    if not os.path.exists(img_path):
        response = requests.get(img_url)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download image {img_id}")

# Prepare subset annotations
# 1. Detection annotations subset
# detection_subset_annotations = {
#     "images": [img for img in detection_coco.dataset['images'] if img['id'] in selected_image_ids],
#     "annotations": [ann for ann in detection_coco.dataset['annotations'] if ann['image_id'] in selected_image_ids],
#     "categories": detection_coco.dataset['categories']
# }

# 2. Captions annotations subset
captions_subset_annotations = {
    "images": [img for img in captions_coco.dataset['images'] if img['id'] in selected_image_ids],
    "annotations": [ann for ann in captions_coco.dataset['annotations'] if ann['image_id'] in selected_image_ids],
    "categories": captions_coco.dataset['categories']
}

# # Save detection subset annotations
# with open(output_detection_ann_path, 'w') as f:
#     json.dump(detection_subset_annotations, f)

# Save captions subset annotations
with open(output_captions_ann_path, 'w') as f:
    json.dump(captions_subset_annotations, f)

print("Subset creation completed. Images and annotations for detection and captions have been saved.")
