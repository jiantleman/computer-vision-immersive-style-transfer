import os
import argparse
import numpy as np
from scipy.optimize import minimize

import tensorflow as tf
from skimage import io, img_as_float32, transform
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import backend
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.compat.v1 as tfc
tfc.disable_v2_behavior()

import hyperparameters as hp

# Hyperparameter Constants
IMG_WIDTH = hp.IMG_WIDTH
IMG_HEIGHT = hp.IMG_HEIGHT
IMAGE_NET_MEAN_RGB = hp.IMAGE_NET_MEAN_RGB
CHANNELS = hp.CHANNELS
CONTENT_WEIGHT = hp.CONTENT_WEIGHT
STYLE_WEIGHT = hp.STYLE_WEIGHT
TOTAL_VARIATION_WEIGHT = hp.TOTAL_VARIATION_WEIGHT
LOSS_FACTOR = hp.LOSS_FACTOR
CONTENT_LAYER = hp.CONTENT_LAYER
STYLE_LAYERS = hp.STYLE_LAYERS
ITER_PER_EPOCH = hp.ITER_PER_EPOCH
OPTIMIZER_METHOD = hp.OPTIMIZER_METHOD
EPOCHS = hp.EPOCHS

#====================================================================

# Mean normalization and preprocessing to format required for tensor
def preprocess_image(image):
    image = np.expand_dims(image, axis=0)
    image[:, :, :, 0] -= IMAGE_NET_MEAN_RGB[0]
    image[:, :, :, 1] -= IMAGE_NET_MEAN_RGB[1]
    image[:, :, :, 2] -= IMAGE_NET_MEAN_RGB[2]
    image = image[:, :, :, ::-1]
    return image

def path_to_image(image_path):
    image = io.imread(image_path)
    image = np.asarray(image, dtype="float32")
    return image

#--------------------

# Calculating content loss from feature & combination image
def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

# With weights
def calc_content_loss(output):
    content_image_activations = output[0, :, :, :]
    combination_image_activations = output[2, :, :, :]
    return CONTENT_WEIGHT * content_loss(content_image_activations, combination_image_activations)

#--------------------

# Calculating style loss from feature & combination image
def style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    numerator = backend.sum(backend.square(style - combination))
    denominator = 4.0 * (CHANNELS ** 2) * ((IMG_HEIGHT * IMG_WIDTH) ** 2)
    return numerator/denominator

# With weights
def calc_style_loss(output, num_layers):
    style_image_activations = output[1, :, :, :]
    combination_image_activations = output[2, :, :, :]
    return (STYLE_WEIGHT/num_layers) * style_loss(style_image_activations, combination_image_activations)

# Using the gram matrix equation
def gram_matrix(image):
    image = backend.batch_flatten(backend.permute_dimensions(image, (2, 0, 1)))
    return backend.dot(image, backend.transpose(image))
    # numerator_matrix = tf.einsum('bijc,bijd->bcd', image, image)
    # image_dimension = tf.shape(image)
    # denominator = image_dimension[1] * image_dimension[2]
    # return numerator_matrix/denominator

#--------------------

# Calculate total variation loss of an image
def total_variation_loss(image):
    height_variation = backend.square(image[:, :IMG_HEIGHT-1, :IMG_WIDTH-1, :] - image[:, :IMG_HEIGHT-1, 1:, :])
    width_variation = backend.square(image[:, :IMG_HEIGHT-1, :IMG_WIDTH-1, :] - image[:, 1:, :IMG_WIDTH-1, :])
    return backend.sum(backend.pow(height_variation + width_variation, LOSS_FACTOR))

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
                        required=False,
                        help='Path to the content image.')
    parser.add_argument('--style_image_path',
                        type=str,
                        default=os.getcwd() + '/data/1_style.jpg',
                        required=False,
                        help='Path to the style image.')
    parser.add_argument('--output_image_path',
                        type=str,
                        default=os.getcwd() + '/results/output.jpg',
                        required=False,
                        help='Path to the output image.')

    args = parser.parse_args()

    # Get images and preprocess
    output_image_path = args.output_image_path
    if not os.path.exists('results'):
        os.makedirs('results')

    content_image = path_to_image(args.content_image_path)
    ORIGINAL_IMG_HEIGHT = content_image.shape[0]
    ORIGINAL_IMG_WIDTH = content_image.shape[1]
    style_image = path_to_image(args.style_image_path)
    style_image = transform.resize(style_image, (IMG_HEIGHT, IMG_WIDTH))
    processed_style_image = preprocess_image(style_image)
    print("=====================Style image resized and preprocessed=====================")


    style_image = backend.variable(processed_style_image)




    final_output_image = np.zeros((ORIGINAL_IMG_HEIGHT, ORIGINAL_IMG_WIDTH, 3), dtype=np.uint8)

    for w in range(int(ORIGINAL_IMG_WIDTH/IMG_WIDTH)):
        for h in range(int(ORIGINAL_IMG_HEIGHT/IMG_HEIGHT)):
            content_image_patch = content_image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH]
            processed_content_image = preprocess_image(content_image_patch)

            
            ### Haven't touched anything below this ###

            # Combining images into tensor
            content_image = backend.variable(processed_content_image)
            combination_image = backend.placeholder((1, IMG_HEIGHT, IMG_WIDTH, 3))
            input_tensor = backend.concatenate([content_image, style_image, combination_image], axis=0)

            # Load VGG model
            model = VGG19(input_tensor=input_tensor, include_top=False, weights="imagenet")

            # Freeze weights
            for layer in model.layers:
                layer.trainable = False
            
            model.summary()
            print("=====================VGG model set up=====================")
            
            # Forming the name-output dictionary for each layer
            cnn_layers = dict([(layer.name, layer.output) for layer in model.layers])

            # Content loss
            content_output = cnn_layers[CONTENT_LAYER]
            loss = calc_content_loss(content_output)
            # Style loss
            num_style_layers = len(STYLE_LAYERS)
            for layer_name in STYLE_LAYERS:
                style_layer_output = cnn_layers[layer_name]
                loss += calc_style_loss(style_layer_output, num_style_layers)
            # Total variation loss
            loss += calc_total_variation_loss(combination_image)
            
            print("=====================All losses set-up=====================")

            gradient = backend.gradients(loss, [combination_image])

            print("=====================Gradient tensor set-up=====================")

            loss_output = [loss]
            gradients_output = [gradient]
            
            class Evaluator:

                def loss(self, x):
                    x = x.reshape((1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
                    get_loss = backend.function([combination_image], loss_output)

                    [cur_loss] = get_loss([x])
                    return cur_loss

                def gradients(self, x):
                    x = x.reshape((1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
                    get_gradients = backend.function([combination_image], gradients_output)

                    [cur_gradients] = get_gradients([x])
                    cur_gradients = np.array(cur_gradients).flatten().astype("float64")
                    return cur_gradients
                
            evaluator = Evaluator()

            # Initialize with the fixed content image to get deterministic results
            generated_vals = processed_content_image
            
            for i in range(EPOCHS):
                optimize_result = minimize(
                    evaluator.loss,
                    generated_vals.flatten(),
                    method=OPTIMIZER_METHOD,
                    jac=evaluator.gradients,
                    options={'maxiter': ITER_PER_EPOCH})
                generated_vals = optimize_result.x
                loss = optimize_result.fun
                print("Epoch %d completed with loss %d" % (i, loss))
            
            generated_vals = generated_vals.reshape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
            generated_vals = generated_vals[:, :, ::-1]
            generated_vals += IMAGE_NET_MEAN_RGB
            output_image = np.clip(generated_vals, 0, 255).astype("uint8")

            ### Till here ###

            final_output_image[h*IMG_HEIGHT:(h+1)*IMG_HEIGHT, w*IMG_WIDTH:(w+1)*IMG_WIDTH, :] = output_image

    # Save generated image
    plt.imsave(output_image_path, final_output_image)
    print("=====================Done=====================")

    
main()
