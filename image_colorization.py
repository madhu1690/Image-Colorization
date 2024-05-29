import numpy as np
import cv2 as cv
import os

# Paths to the required files
frame_path = r"C:\Users\s\OneDrive\Pictures\girl-2529907_640.jpg"
prototxt_path = r"C:\Users\s\Downloads\colorization_deploy_v2.prototxt"
caffe_model_path = r"C:\Users\s\Downloads\colorization_release_v2.caffemodel"
pts_npy_path = r"C:\Users\s\Downloads\pts_in_hull.npy"
output_path = r"D:\c++\python image project\mkdir models\result.png"

# Load the input image
frame = cv.imread(frame_path)
if frame is None:
    raise FileNotFoundError(f"Image file not found at {frame_path}")

# Load the cluster centers
numpy_file = np.load(pts_npy_path)

# Load the pre-trained Caffe model
Caffe_net = cv.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

# Prepare the cluster centers
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)

# Assign cluster centers to the network
Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# Set the input size for the network
input_width = 224
input_height = 224

# Convert the input image to float and RGB format
rgb_img = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)

# Convert the RGB image to LAB color space
lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
l_channel = lab_img[:, :, 0]  # Extract the L channel

# Resize the L channel to network input size and normalize
l_channel_resize = cv.resize(l_channel, (input_width, input_height))
l_channel_resize -= 50  # Normalize

# Set the input to the network
Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))

# Forward pass to get the predicted AB channels
ab_channel = Caffe_net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the AB channels to the original image size
(original_height, original_width) = rgb_img.shape[:2]
ab_channel_us = cv.resize(ab_channel, (original_width, original_height))

# Concatenate the L channel with the predicted AB channels
lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_channel_us), axis=2)

# Convert LAB color space back to BGR
bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)

# Save the output image
cv.imwrite(output_path, (bgr_output * 255).astype(np.uint8))

print(f"Colorized image saved to {output_path}")

