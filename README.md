# CL-RTDETR-DIFFUSION
Continual Learning Object Detection with Diffusion Data Replay
# Download data
```bash
mkdir coco && cd coco 
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017 && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017 && rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017 && rm annotations_trainval2017.zip
cd ..
# Note: Change your COCO path on `configs/dataset/coco_detection.yml`
# Install
```
# Install python environment
```bash
conda create -n tan python=3.10 
pip install -r requirements.txt
```
# Traing
```bash
# NOTE: change task 0, task 1 in configs/cl_pipeline.yml and configs\rtdetr\include\dataloader.yml 
python scrirps/train.py
```
# Eval
```bash
python scrirps/train.py -c configs\rtdetr\rtdetr_r50vd_coco.yml -r path/to/checkpoint --test-only
```
# AUTHOR
NGO VAN TAN 20210769