import os
import argparse
import numpy as np
import tensorflow as tf
from skimage import io, img_as_float32, transform
import matplotlib.pyplot as plt
from tensorflow.keras import backend
from tensorflow.keras.applications import vgg19 as V
from tensorflow.keras.applications.vgg19 import VGG19

# Hyperparameter Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMAGE_NET_MEAN_RGB = [103.939,116.779,123.68]
CONTENT_WEIGHT = 0.025
STYLE_WEIGHT = 1.0
TOTAL_VARIATION_WEIGHT = 1.0
LOSS_FACTOR = 1.25
CHANNELS = 3
CONTENT_LAYER = 'block5_conv2'
STYLE_LAYER = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

#====================================================================

# Mean normalization and preprocessing to format required for tensor
def preprocess_image(image):
    image = np.expand_dims(image, axis=0)
    # image = V.preprocess_input(image)
    image[:, :, :, 0] -= IMAGE_NET_MEAN_RGB[0]
    image[:, :, :, 1] -= IMAGE_NET_MEAN_RGB[1]
    image[:, :, :, 2] -= IMAGE_NET_MEAN_RGB[2]
    image = image[:, :, :, ::-1]
    return image

#--------------------

# Calculating content loss from feature & combination image
def content_loss(image, combination):
    return backend.sum(backend.square(combination - image))

# With weights
def calc_content_loss(output):
    image = output[0, :, :, :]
    combination = output[2, :, :, :]
    return CONTENT_WEIGHT * content_loss(image, combination)

#--------------------

# Calculating style loss from feature & combination image
def style_loss(image, combination):
    image = gram_matrix(image)
    combination = gram_matrix(combination)
    numerator = backend.sum(backend.square(image - combination))
    denominator = 4.0 * (CHANNELS ** 2) * ((IMG_HEIGHT * IMG_WIDTH) ** 2)
    return numerator/denominator

# With weights
def calc_style_loss(output, num_output):
    image = output[1, :, :, :]
    combination = output[2, :, :, :]
    return (STYLE_WEIGHT/num_output) * style_loss(image, combination)

# Using the gram matrix equation
def gram_matrix(image):
    numerator_matrix = tf.einsum('bijc,bijd->bcd', image, image)
    image_dimension = tf.shape(image)
    denominator = image_dimension[1] * image_dimension[2]
    return numerator_matrix/denominator

#--------------------

# Calculate total variation loss of an image
def total_variation_loss(image):
    height_variation = image[:, :IMG_HEIGHT-1, :IMG_WIDTH-1, :] - image[:, :IMG_HEIGHT-1, 1:, :]
    width_variation = image[:, :IMG_HEIGHT-1, :IMG_WIDTH-1, :] - image[:, 1:, :IMG_WIDTH-1, :]
    return backend.sum(backend.pow(backend.square(height_variation) + backend.square(width_variation), LOSS_FACTOR))

# With weights
def calc_total_variation_loss(image):
    return TOTAL_VARIATION_WEIGHT * total_variation_loss(image)

#====================================================================


def main():
    # Command-line parsing
    parser = argparse.ArgumentParser(description="Style Transer with CNN")
    parser.add_argument('--content_image_path',
                        type=str,
                        default=os.getcwd() + '/data/1_content.jpg',
                        help='Path to the content image.')
    parser.add_argument('--style_image_path',
                        type=str,
                        default=os.getcwd() + '/data/1_style.jpg',
                        help='Path to the style image.')
    args = parser.parse_args()

    # Get images and preprocess
    content_image_path = args.content_image_path
    style_image_path = args.style_image_path    

    content_image = img_as_float32(io.imread(content_image_path))
    content_image = transform.resize(content_image,(IMG_WIDTH, IMG_HEIGHT))
    #content_image = preprocess_image(content_image)
    
    style_image = img_as_float32(io.imread(style_image_path))
    style_image = transform.resize(style_image,(IMG_WIDTH, IMG_HEIGHT))
    #style_image = preprocess_image(style_image)
    print("Images resized")

    # Combining images into tensor
    content_image = backend.variable(preprocess_image(content_image))
    style_image = backend.variable(preprocess_image(style_image))
    combination_image = backend.placeholder((1, IMG_HEIGHT, IMG_WIDTH, 3))
    input_tensor = backend.concatenate([content_image,style_image,combination_image], axis=0)

    # Load VGG model
    model = VGG19(input_tensor=input_tensor, include_top=False)
    print("VGG model set up")

    # Forming the name-output dictionary for each layer
    cnn_layers = dict([(layer.name, layer.output) for layer in model.layers])

    loss = backend.variable(0.0)
    # Content loss
    content_output = cnn_layers[CONTENT_LAYER]
    loss.assign_add(calc_content_loss(content_output))
    # Style loss
    num_layer = len(STYLE_LAYER)
    for name in STYLE_LAYER:
        style_output = cnn_layers[name]
        loss.assign_add(calc_style_loss(style_output, num_layer))
    # Total variation loss
    loss.assign_add(calc_total_variation_loss(combination_image))
    
    print("All loss calculated")
        


    


main()
