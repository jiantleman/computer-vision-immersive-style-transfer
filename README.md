# Immersive 360° Style Transfer
In this project, we first used image representations derived from Convolutional Neural Networks to combine the semantic content of an image with the style of another image, based on the technique developed by Gatys et al. Next, we scaled the style transfer technique up from 512 x 512 pixels in the original paper and applied it to 6000 x 3000 pixels to create immersive 360° panorama images. Due to GPU memory limitations, the maximum image size that we were able to run the algorithm on was 1500 x 750 pixels. We thereby ran style transfer on smaller, divided portions of the content image before performing post-processing to combine them seamlessly into a full-sized image. 

To view the final images, please visit our panorama viewer here: https://miku-suga.github.io/cs1430-final-project/visualizer.html

## Code organization
### Results
All generated images can be found under the results/ folder.

'*_output.jpg' images are the final stitched and processed images.

'*_output_base.jpg' images are the smaller 1500x750px sized stylised images.

'*_output_large.jpg' images are the stiched together, but unprocessed 4x4 images.

### Data
The original content and style images can be viewed under the data/ folder.

### Style Transfer
The code that was used to perform style transfer can be found in main.py and the hyperparameters used can be found in hyperparameters.py

### Image post-processing
Image post-processing is carried out by stitching.py with some hyperparameters taken from hyperparameters.py

### Visualizer
All code for the visualizer is self-contained within visualizer.html
