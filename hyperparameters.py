# Image preprocessing variables
IMG_WIDTH = 300
IMG_HEIGHT = 150
IMAGE_NET_MEAN_RGB = [103.939,116.779,123.68]
CHANNELS = 3
SCALE = 2

# Loss hyperparameters
CONTENT_WEIGHT = 0.005
STYLE_WEIGHT = 5.0
TOTAL_VARIATION_WEIGHT = 1.0
VARIATION_FACTOR = 1.25

# Neural Network hyperparameters
CONTENT_LAYER = 'block4_conv2'
STYLE_LAYERS = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

# Optimizer hyperparameters
ITER_PER_EPOCH = 20
OPTIMIZER_METHOD = 'L-BFGS-B'
EPOCHS = 10
