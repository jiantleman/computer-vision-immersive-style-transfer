# Image preprocessing variables
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMAGE_NET_MEAN_RGB = [103.939,116.779,123.68]
CHANNELS = 3

# Loss hyperparameters
CONTENT_WEIGHT = 0.25
STYLE_WEIGHT = 5.0
TOTAL_VARIATION_WEIGHT = 1.0
LOSS_FACTOR = 1.25

# Neural Network hyperparameters
CONTENT_LAYER = 'block4_conv2'
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Optimizer hyperparameters
ITER_PER_EPOCH = 100
OPTIMIZER_METHOD = 'L-BFGS-B'
EPOCHS = 10
