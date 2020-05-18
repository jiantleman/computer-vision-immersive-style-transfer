import os
import argparse
import numpy as np
from skimage import io, transform
from PIL import Image

import hyperparameters as hp

# Global Constants
IMG_WIDTH = hp.IMG_WIDTH
IMG_HEIGHT = hp.IMG_HEIGHT
SCALE = hp.SCALE

NUM_PASSES = 1
FIX_WIDTH = 200
FIX_FACTOR = 1/FIX_WIDTH


# Parse command-line arguments
def parse_args():
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
    return parser.parse_args()


# Preprocess images
def get_scaled_image(image_path):
    image = io.imread(image_path)
    image = np.asarray(image, dtype="float32")
    image = transform.resize(image, (SCALE*IMG_HEIGHT, SCALE*IMG_WIDTH))
    return image


# Normalize the target value using the base value
def normalize(target, base):
    # Avoid destructive normalization
    target_copy = np.copy(target)
    target_copy = (target_copy - np.mean(target)) / np.std(target)
    target_copy = target_copy * np.std(base) + np.mean(base)
    return target_copy


# Blend the target value with the adjustment value based on the multiplier
def mixin(target, adjustment, multiplier):
    cur_factor = -np.log(FIX_FACTOR * multiplier) / 2
    target += cur_factor * adjustment
    return target / (1 + cur_factor)


def main():
    # Get images
    output_image_path = ARGS.output
    large_image_path = ARGS.large
    base_image_path = ARGS.base

    image = get_scaled_image(large_image_path)
    large_base = get_scaled_image(base_image_path)

    for h in range(SCALE):
        for w in range(SCALE):
            cur_quad = image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT,
                             w*IMG_WIDTH:(w+1)*IMG_WIDTH,
                             :]
            base_quad = large_base[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT,
                                   w*IMG_WIDTH:(w+1)*IMG_WIDTH,
                                   :]
            # Renormalize Quadrant
            cur_quad = normalize(cur_quad, base_quad)

            # Renormalize each horizontal band in the quadrant
            for i in range(IMG_HEIGHT):
                cur_band = cur_quad[i, :, :]
                base_band = base_quad[i, :, :]
                cur_quad[i, :, :] = normalize(cur_band, base_band)

            # Renormalize each vertical band in the quadrant
            for i in range(IMG_WIDTH):
                cur_band = cur_quad[:, i, :]
                base_band = base_quad[:, i, :]
                cur_quad[:, i, :] = normalize(cur_band, base_band)

            # Feather blend edges
            for _ in range(NUM_PASSES):
                for i in range(1, FIX_WIDTH):
                    cur_quad[:, i-1, :] = mixin(cur_quad[:, i-1, :],
                                                base_quad[:, i-1, :],
                                                i)
                    cur_quad[:, -i, :] = mixin(cur_quad[:, -i, :],
                                               base_quad[:, -i, :],
                                               i)
                    cur_quad[i-1, ...] = mixin(cur_quad[i-1, ...],
                                               base_quad[i-1, ...],
                                               i)
                    cur_quad[-i, ...] = mixin(cur_quad[-i, ...],
                                              base_quad[-i, ...],
                                              i)

            image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT,
                  w*IMG_WIDTH:(w+1)*IMG_WIDTH,
                  :] = cur_quad
    # Save image
    image = np.clip(image, 0, 255).astype("uint8")
    image = Image.fromarray(image)
    image.save(output_image_path)


ARGS = parse_args()

main()
