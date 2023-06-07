

import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import h5py
import os
from vggnet import VGGNet

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
h5py.get_config().track_order=True
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Set input and output paths
input_path = "/home/raja/Desktop/MICCAI/data/lysto-test-data/"
output_path = "output_images"

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Load the first 10 images from input directory
numofimages = 10
image_files = os.listdir(input_path)[:]


# Set mask generation parameters
mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch= 64,
    pred_iou_thresh= 0.90,
    stability_score_thresh= 0.96,
    stability_score_offset= 1.0,
    box_nms_thresh= 0.7,
    crop_n_layers= 0,
    crop_nms_thresh= 0.5,
    crop_overlap_ratio= 0,  #5 / 1500,
    crop_n_points_downscale_factor= 1,
    point_grids= None,
    min_mask_region_area= 200,
)




experiment_name='lysto-test-result'


outputdir=os.path.join('test-data-output',experiment_name)
if(os.path.isdir(outputdir)):
    print('Warning: Directory with experiment name %s exists. Preferable to have an empty directory to avoid stale files')
else:
    os.makedirs(outputdir)


from PIL import Image

# Process each image
for i, filename in enumerate(image_files):
    # Load image
    image = cv2.imread(os.path.join(input_path, filename))

    # Convert image to RGB format
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate masks and crop images
    masks = mask_generator_.generate(image)

    imge = Image.open(os.path.join(input_path, filename))


    bounding_boxes_b = []
    for j, mask in enumerate(masks):
        bbox = mask["bbox"]
        xmin, ymin, w, h = bbox
        xmax, ymax = xmin + w, ymin + h
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax) # convert to integers
        temp_img = imge.copy()
        temp_mask = mask['segmentation'].copy()

        # convert mask to uint8
        temp_mask = temp_mask.astype(np.uint8)

        # multiply image with mask
        masked_img = temp_img * temp_mask[..., np.newaxis]
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        cropped_img = masked_img[ymin:ymax, xmin:xmax]
        if 15*15 < cropped_img.shape[0] * cropped_img.shape[1] < 41*41:
            filename_without_extension = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(outputdir, f"{filename_without_extension}_cropped_{j}.jpg"), cropped_img)
