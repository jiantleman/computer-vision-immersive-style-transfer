import os
import argparse
import numpy as np
from skimage import io, img_as_float32, transform, filters
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image, ImageFilter

NUM_PASSES = 1
FIX_WIDTH = 200
FIX_FACTOR = 1/FIX_WIDTH
IMG_WIDTH = 1500
IMG_HEIGHT = 750
SCALE = 4
HORIZONTAL_PASSES = 5
VERTICAL_PASSES = 3

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
image = transform.resize(image,(SCALE*IMG_HEIGHT, SCALE*IMG_WIDTH))

base = io.imread(base_image_path)
base = np.asarray(base, dtype="float32")
large_base = transform.resize(base,(SCALE*IMG_HEIGHT, SCALE*IMG_WIDTH))

for h in range(SCALE):
    for w in range(SCALE):
        cur_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:]
        base_quad = large_base[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:]
        print(h, w)
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
        cur_quad = (cur_quad - np.mean(cur_quad))/np.std(cur_quad)
        cur_quad = cur_quad * np.std(base_quad) + np.mean(base_quad)
    
        image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:] = cur_quad


# Normalize edges along bands
for _ in range(HORIZONTAL_PASSES):
    for h in range(SCALE - 1):
        for w in range(SCALE):
            top_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH, :]
            bot_quad = image[(h+1)*IMG_HEIGHT:(h+2)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH, :]
            for i in range(1, FIX_WIDTH):
                cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
                top_band = top_quad[-i, :, :]
                bot_band = bot_quad[i-1, :, :]
                top_mean = np.mean(top_band)
                top_std = np.std(top_band)
                bot_mean = np.mean(bot_band)
                bot_std = np.std(bot_band)
                top_band = (top_band - top_mean) / top_std
                top_band = top_band * bot_std + bot_mean
                bot_band = (bot_band - bot_mean) / bot_std
                bot_band = bot_band * top_std + top_mean
                top_quad[-i, :, :] += cur_factor * top_band
                bot_quad[i-1, :, :] += cur_factor * bot_band
                top_quad[-i, :, :] /= 1 + cur_factor
                bot_quad[i-1, :, :] /= 1 + cur_factor
            
for _ in range(VERTICAL_PASSES):
    for h in range(SCALE):
        for w in range(SCALE - 1):
            left_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:]
            right_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, (w+1)*IMG_WIDTH:(w+2)*IMG_WIDTH, :]
            for i in range(1, FIX_WIDTH):
                cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
                left_band = left_quad[:, -i, :]
                right_band = right_quad[:, i-1, :]
                left_mean = np.mean(left_band)
                left_std = np.std(left_band)
                right_mean = np.mean(right_band)
                right_std = np.std(right_band)
                left_band = (left_band - left_mean) / left_std
                left_band = left_band * right_std + right_mean
                right_band = (right_band - right_mean) / right_std
                right_band = right_band * left_std + left_mean
                left_quad[:, -i, :] += cur_factor * left_band
                right_quad[:, i-1, :] += cur_factor * right_band
                left_quad[:, -i, :] /= 1 + cur_factor
                right_quad[:, i-1, :] /= 1 + cur_factor 

# Fix seam
#image = image[:, 10:-10, :]
'''
image_copy = np.copy(image)
for i in range(1, FIX_WIDTH):
    cur_factor = (np.exp(1 - FIX_FACTOR * i)-1)
    image[:, i-1, :] += cur_factor * image_copy[:, -i, :]
    image[:, -i, :] += cur_factor * image_copy[:, i-1, :]
    image[:, i-1, :] /= 1 + cur_factor
    image[:, -i, :] /= 1 + cur_factor
'''

# And renormalize everything band by band
for h in range(SCALE):
    for w in range(SCALE):
        cur_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:]
        base_quad = large_base[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:]
        for i in range(IMG_HEIGHT):
            cur_band = cur_quad[i, :, :]

            base_band = base_quad[i, :, :]

            cur_band = (cur_band - np.mean(cur_band)) / np.std(cur_band)
            cur_band = cur_band * np.std(base_band) + np.mean(base_band)
            cur_quad[i, :, :] = cur_band

        for i in range(IMG_WIDTH):
            cur_band = cur_quad[:, i, :]
            base_band = base_quad[:, i, :]
            cur_band = (cur_band - np.mean(cur_band)) / np.std(cur_band)
            cur_band = cur_band * np.std(base_band) + np.mean(base_band)

            cur_quad[:, i, :] = cur_band
            
        image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH,:] = cur_quad


# Save image
image =  np.clip(image, 0, 255).astype("uint8")
image = Image.fromarray(image)
image.save(output_image_path)
