import os
import argparse
import numpy as np
from skimage import io, img_as_float32, transform, filters
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image, ImageFilter

NUM_PASSES = 2
FIX_WIDTH = 5
FIX_FACTOR = 1/FIX_WIDTH
IMG_WIDTH = 1500
IMG_HEIGHT = 750
SEAM_WIDTH = 1

parser = argparse.ArgumentParser(description="Style Transer with CNN")
parser.add_argument('--input',
                    type=str,
                    default=os.getcwd() + '/results/output.jpg',
                    required=False,
                    help='Path to the original image.')
parser.add_argument('--output',
                    type=str,
                    default=os.getcwd() + '/results/fixed.jpg',
                    required=False,
                    help='Path to the output image.')

args = parser.parse_args()

output_image_path = args.output
if not os.path.exists('results'):
    os.makedirs('results')
input_image_path = args.input

image = io.imread(input_image_path)
image = np.asarray(image, dtype="float32")

# Blend seam
image = image[:, SEAM_WIDTH:-SEAM_WIDTH, :]
image_copy = np.copy(image)

for _ in range(NUM_PASSES):
    for i in range(1, FIX_WIDTH):
        cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
        image[:, i-1, :] += cur_factor * image_copy[:, -i, :]
        image[:, -i, :] += cur_factor * image_copy[:, i-1, :]
        image[:, i-1, :] /= 1 + cur_factor
        image[:, -i, :] /= 1 + cur_factor

# Normalize seam
for _ in range(1):
    for i in range(1, FIX_WIDTH):
        cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
        left_band = image[:, -i, :]
        right_band = image[:, i-1, :]
        left_norms = np.linalg.norm(left_band, axis=0)
        right_norms = np.linalg.norm(right_band, axis=0)
        left_band = (left_band * right_norms) / left_norms
        right_band = (right_band * left_norms) / right_norms
        image[:, -i, :] += cur_factor * left_band
        image[:, i-1, :] += cur_factor * right_band
        image[:, -i, :] /= 1 + cur_factor
        image[:, i-1, :] /= 1 + cur_factor 


# Save image
image =  np.clip(image, 0, 255).astype("uint8")
image = Image.fromarray(image)
image.save(output_image_path)
