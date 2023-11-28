import numpy as np
import cv2 as cv
import os.path
frame = cv.imread("C:\\Users\\s\\OneDrive\\Pictures\\girl-2529907_640.jpg")
prototxt_path = r"C:\Users\s\Downloads\colorization_deploy_v2.prototxt"
caffe_model_path = r"C:\Users\s\Downloads\colorization_release_v2.caffemodel"
pts_npy_path = r"C:\Users\s\Downloads\pts_in_hull.npy"

# Load the files
numpy_file = np.load(pts_npy_path)
Caffe_net = cv.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)
Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
input_width = 224
input_height = 224

rgb_img = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
l_channel = lab_img[:,:,0]

l_channel_resize = cv.resize(l_channel, (input_width, input_height))
l_channel_resize -= 50
Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))
ab_channel = Caffe_net.forward()[0,:,:,:].transpose((1,2,0))

(original_height,original_width) = rgb_img.shape[:2]
ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
lab_output = np.concatenate((l_channel[:,:,np.newaxis],ab_channel_us),axis=2)
bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0 , 1)
cv.imwrite(r"D:\c++\python image project\mkdir models\result.png", (bgr_output*255).astype(np.uint8))
