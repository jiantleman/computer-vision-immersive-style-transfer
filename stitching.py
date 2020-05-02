import os
import argparse
import numpy as np
from skimage import io, img_as_float32, transform, filters
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image, ImageFilter

NUM_PASSES = 1
FIX_WIDTH = 100
FIX_FACTOR = 1/FIX_WIDTH
IMG_WIDTH = 1500
IMG_HEIGHT = 750

parser = argparse.ArgumentParser(description="Style Transer with CNN")
parser.add_argument('--large',
                    type=str,
                    default=os.getcwd() + '/results/output.jpg',
                    required=False,
                    help='Path to the high resolution image.')
parser.add_argument('--base',
                    type=str,
                    default=os.getcwd() + '/data/1_base.jpg',
                    required=False,
                    help='Path to the low resolution base image.')
parser.add_argument('--output',
                    type=str,
                    default=os.getcwd() + '/results/stitched.jpg',
                    required=False,
                    help='Path to the output image.')

args = parser.parse_args()

output_image_path = args.output
if not os.path.exists('results'):
    os.makedirs('results')
large_image_path = args.large
base_image_path = args.base

image = io.imread(large_image_path)
image = np.asarray(image, dtype="float32")
image = transform.resize(image,(2*IMG_HEIGHT, 2*IMG_WIDTH))

base = io.imread(base_image_path)
base = np.asarray(base, dtype="float32")
large_base = transform.resize(base,(2*IMG_HEIGHT, 2*IMG_WIDTH))

adj = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for (h, w) in adj:
    cur_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:]
    base_quad = large_base[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:]

    # Feather blend edges
    for _ in range(NUM_PASSES):
        for i in range(1, FIX_WIDTH):
            cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
            cur_quad[:, i-1, :] += cur_factor * base_quad[:, i-1, :]
            cur_quad[:, -i, :] += cur_factor * base_quad[:, -i, :]
            cur_quad[:, i-1, :] /= 1 + cur_factor
            cur_quad[:, -i, :] /= 1 + cur_factor
            cur_quad[i-1, :, :] += cur_factor * base_quad[i-1, :, :]
            cur_quad[-i, :, :] += cur_factor * base_quad[-i, :, :]
            cur_quad[i-1, :, :] /= 1 + cur_factor
            cur_quad[-i, :, :] /= 1 + cur_factor

    # Renormalize Quadrant
    norm = np.linalg.norm(cur_quad)
    cur_quad /= norm
    
    image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:] = cur_quad

base_norm = np.linalg.norm(base)
image *= base_norm

# Normalize edges along bands
for _ in range(5):
    for w in range(2):
        top_quad = image[:IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH, :]
        bot_quad = image[IMG_HEIGHT:, w*IMG_WIDTH:(w+1)*IMG_WIDTH, :]
        for i in range(1, FIX_WIDTH):
            cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
            top_band = top_quad[-i, :, :]
            bot_band = bot_quad[i-1, :, :]
            top_norms = np.linalg.norm(top_band, axis=0)
            bot_norms = np.linalg.norm(bot_band, axis=0)
            top_band = (top_band * bot_norms) / top_norms
            bot_band = (bot_band * top_norms) / bot_norms
            top_quad[-i, :, :] += cur_factor * top_band
            bot_quad[i-1, :, :] += cur_factor * bot_band
            top_quad[-i, :, :] /= 1 + cur_factor
            bot_quad[i-1, :, :] /= 1 + cur_factor
            
for _ in range(1):
    for h in range(2):
        left_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, :IMG_WIDTH,:]
        right_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, IMG_WIDTH:, :]
        for i in range(1, FIX_WIDTH):
            cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
            left_band = left_quad[:, -i, :]
            right_band = right_quad[:, i-1, :]
            left_norms = np.linalg.norm(left_band, axis=0)
            right_norms = np.linalg.norm(right_band, axis=0)
            left_band = (left_band * right_norms) / left_norms
            right_band = (right_band * left_norms) / right_norms
            left_quad[:, -i, :] += cur_factor * left_band
            right_quad[:, i-1, :] += cur_factor * right_band
            left_quad[:, -i, :] /= 1 + cur_factor
            right_quad[:, i-1, :] /= 1 + cur_factor 

# Fix seam
image = image[:, 10:-10, :]
image_copy = np.copy(image)
for i in range(1, FIX_WIDTH):
    cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
    image[:, i-1, :] += cur_factor * image_copy[:, -i, :]
    image[:, -i, :] += cur_factor * image_copy[:, i-1, :]
    image[:, i-1, :] /= 1 + cur_factor
    image[:, -i, :] /= 1 + cur_factor

# Save image
image =  np.clip(image, 0, 255).astype("uint8")
image = Image.fromarray(image)
image.save(output_image_path)
