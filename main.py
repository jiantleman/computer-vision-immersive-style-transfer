import argparse
import numpy as np
from skimage import io, img_as_float32, transform
import matplotlib.pyplot as plt
from tensorflow.keras import backend
from tensorflow.keras.applications.vgg19 import VGG19

# Command-line parsing
parser = argparse.ArgumentParser(description="Style Transer with CNN")
parser.add_argument('content_image_path', type=str, help='Path to the content image.')
parser.add_argument('style_image_path', type=str, help='Path to the style image.')
args = parser.parse_args()
content_image_path = args.content_image_path
style_image_path = args.style_image_path

# Constants
WIDTH = 128
HEIGHT = 128
IMAGE_NET_MEAN_RGB = [103.939,116.779,123.68]

content_image = img_as_float32(io.imread(content_image_path))
content_image = transform.resize(content_image,(WIDTH,HEIGHT))
style_image = img_as_float32(io.imread(style_image_path))
style_image = transform.resize(style_image,(WIDTH,HEIGHT))

# Mean normalization and preprocessing to format required for tensor
def preprocess_image(image):
    image = np.expand_dims(image, axis=0)
    image[:, :, :, 0] -= IMAGE_NET_MEAN_RGB[0]
    image[:, :, :, 1] -= IMAGE_NET_MEAN_RGB[1]
    image[:, :, :, 2] -= IMAGE_NET_MEAN_RGB[2]
    image = image[:, :, :, ::-1]

# Combining images into tensor
content_image = backend.variable(preprocess_image(content_image))
style_image = backend.variable(preprocess_image(style_image))
combination_image = backend.placeholder((1, WIDTH, HEIGHT, 3))
input_tensor = backend.concatenate([content_image,style_image,combination_image], axis=0)
model = VGG19(input_tensor=input_tensor, include_top=False)

print("complete")