from PIL import Image
import cv2
import numpy as np
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import os
import inflect
from collections import Counter
from tqdm import tqdm
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
class DiffusionSD(CocoDetection):
    def __getitem__(self, index: int):
        id = self.ids[index]
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, path
def replace_small_objects(origin_images, controlnet_images, annotations, threshold=96*96):
    output_images = []
    for origin_image, controlnet_image, annotation in zip(origin_images, controlnet_images, annotations):
        output_image = controlnet_image.copy()
        output_image = output_image.resize(origin_image.size)
        output_image = np.array(output_image)
        origin_image = np.array(origin_image)
        for anno in annotation:
            bbox = anno['bbox']  # COCO bbox format: [x, y, width, height]
            x, y, w, h = map(int, bbox)
            area = w * h
            
            if area < threshold:
            # Replace the region in the output image with the original
                output_image[y:y+h, x:x+w] = origin_image[y:y+h, x:x+w]
        output_image = Image.fromarray(output_image)
        output_images.append(output_image)
    return output_images
def create_canny_images(images):
    canny_images = []
    for image in images:
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = Image.fromarray(image)
        canny_images.append(image)
    return canny_images
def create_prompts(targets):
    positive_prompts = []
    p = inflect.engine()
    for target in targets:
        prompt = " a realistic, detailed photo of "
        ids = [obj['category_id'] for obj in target]
        names = [mscoco_category2name[id] for id in ids]
        names = Counter(names)
        for name, count in names.items():
            if count > 1 and count < 5:
                prompt += p.number_to_words(count) + " " + p.plural(name) + ", "
            elif count > 5:
                prompt += p.plural(name) + ", "
            else:
                prompt += name + ", "
        positive_prompts.append(prompt)
        negative_prompt = ['(deformed , semi-realistic, cgi, render, sketch, cartoon, drawing, anime:1.4), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly'] * len(positive_prompts)
    return positive_prompts, negative_prompt
mscoco_category2name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}
if __name__ == "__main__":

    controlnet_conditioning_scale = 0.8

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
    ).to('cuda')
    
    buffer_path = "/workspace/CL-RTDETR-DIFFUSION/buffer"
    buffer_ann_file = "/workspace/CL-RTDETR-DIFFUSION/buffer/buffer.json"
    buffer_dataset = DiffusionSD(buffer_path, buffer_ann_file)
    buffer_dataloader = DataLoader(buffer_dataset, batch_size=5, shuffle=False, collate_fn= lambda x: tuple(zip(*x)))
    
    buffer_diffusion_dir = "/workspace/CL-RTDETR-DIFFUSION/buffer_diffusion"
    os.makedirs(buffer_diffusion_dir, exist_ok=True)
    for images, targets, paths in tqdm(buffer_dataloader, desc="Diffusion generating: "):
        canny_images = create_canny_images(images)
        positive_prompts, negative_prompts = create_prompts(targets)
        controlnet_outputs = pipe(
            prompt=positive_prompts,negative_prompt=negative_prompts,num_inference_steps=20,guidance_scale=4,num_images_per_prompt=1, image=canny_images
        ).images
        outputs = replace_small_objects(images, controlnet_outputs, targets)
        for output, path in zip(outputs, paths):
            output.save(os.path.join(buffer_diffusion_dir, path))