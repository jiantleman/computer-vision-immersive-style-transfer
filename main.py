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

# Hyperparameter Constants
#try dropping to 54x54
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMAGE_NET_MEAN_RGB = [103.939,116.779,123.68]
CONTENT_WEIGHT = 0.025
STYLE_WEIGHT = 1.0
TOTAL_VARIATION_WEIGHT = 1.0
LOSS_FACTOR = 1.25
CHANNELS = 3
CONTENT_LAYER = 'block5_conv2'
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
EPOCHS = 10

#====================================================================

# Mean normalization and preprocessing to format required for tensor
def preprocess_image(image_path):
    image = img_as_float32(io.imread(image_path))
    image = transform.resize(image,(IMG_WIDTH, IMG_HEIGHT))
    image = np.expand_dims(image, axis=0)
    image[:, :, :, 0] -= IMAGE_NET_MEAN_RGB[0]
    image[:, :, :, 1] -= IMAGE_NET_MEAN_RGB[1]
    image[:, :, :, 2] -= IMAGE_NET_MEAN_RGB[2]
    image = image[:, :, :, ::-1]
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
                        required=True,
                        help='Path to the content image.')
    parser.add_argument('--style_image_path',
                        type=str,
                        default=os.getcwd() + '/data/1_style.jpg',
                        required=True,
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
    content_image_path = args.content_image_path
    style_image_path = args.style_image_path
    content_image = preprocess_image(content_image_path)
    style_image = preprocess_image(style_image_path)    

    # content_image = preprocess_image(content_image_path)
    # content_image = transform.resize(content_image,(IMG_WIDTH, IMG_HEIGHT))
    # style_image = img_as_float32(io.imread(style_image_path))
    # style_image = transform.resize(style_image,(IMG_WIDTH, IMG_HEIGHT))
    
    print("=====================Images resized=====================")

    # Combining images into tensor
    content_image = backend.variable(content_image)
    style_image = backend.variable(style_image)
    combination_image = backend.placeholder((1, IMG_HEIGHT, IMG_WIDTH, 3))
    input_tensor = backend.concatenate([content_image,style_image,combination_image], axis=0)

    # Load VGG model
    model = VGG19(input_tensor=input_tensor, include_top=False, weights="imagenet")

    # Freeze weights
    for layer in model.layers:
        layer.trainable = False
    
    model.summary()
    print("=====================VGG model set up=====================")

    # Get block4_conv2 layer
    #content_layer = model.layers[13]
    

    #layer_outputs = [content_layer.output]
    #activations_model = models.Model(inputs=model.input, outputs=layer_outputs)

    #content_image = backend.variable(content_image)
    #style_image = backend.variable(style_image)
    #with tf.Session() as sess:
        
    #test_image = np.random.normal(0, 1, size=(1, IMG_HEIGHT, IMG_WIDTH, 3)).astype('float32')
    #test_tensor = backend.concatenate([content_image,style_image,test_image], axis=0)


    #activations = activations_model.predict(content_image, steps=1)
    #content_layer_activation = activations[0]
    #plt.matshow(content_layer_activation[:, :, :], cmap='viridis')
    
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

    generated_vals = np.random.uniform(0, 255, (1, IMG_HEIGHT, IMG_WIDTH, 3)) - 128.
    for i in range(EPOCHS):
        optimize_result = minimize(
            evaluator.loss,
            generated_vals.flatten(),
            method='L-BFGS-B',
            jac=evaluator.gradients,
            options={'maxiter': 20})
        generated_vals = optimize_result.x
        loss = optimize_result.fun
        print("Iteration %d completed with loss %d" % (i, loss))
    
    generated_vals = generated_vals.reshape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    generated_vals = generated_vals[:, :, ::-1]
    generated_vals[:, :, 0] += IMAGE_NET_MEAN_RGB[2]
    generated_vals[:, :, 1] += IMAGE_NET_MEAN_RGB[1]
    generated_vals[:, :, 2] += IMAGE_NET_MEAN_RGB[0]
    output_image = np.clip(generated_vals, 0, 255).astype("uint8")

    # Save generated image
    plt.imsave(output_image_path, output_image)

    
main()
