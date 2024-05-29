import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import cv2 as cv

# Updated file paths
prototxt_path = r"C:\Users\s\Downloads\colorization_deploy_v2.prototxt"
caffe_model_path = r"C:\Users\s\Downloads\colorization_release_v2.caffemodel"
pts_npy_path = r"C:\Users\s\Downloads\pts_in_hull.npy"

# Load the files
numpy_file = np.load(pts_npy_path)
Caffe_net = cv.dnn.readNetFromCaffe(prototxt_path, caffe_model_path)

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.title("B&W Image Colorization")
        self.pack(fill=BOTH, expand=1)

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Upload Image", command=self.uploadImage)
        file.add_command(label="Color Image", command=self.color)
        menu.add_cascade(label="File", menu=file)

        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image = None
        self.image_path = None

    def uploadImage(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if not filename:
            return
        self.image_path = filename
        load = Image.open(filename)
        load = load.resize((480, 360), Image.ANTIALIAS)
        self.render = ImageTk.PhotoImage(load)

        if self.image is None:
            self.image = self.canvas.create_image(240, 180, image=self.render)
        else:
            self.canvas.itemconfig(self.image, image=self.render)

    def color(self):
        if not self.image_path:
            return

        frame = cv.imread(self.image_path)

        Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
        Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

        input_width = 224
        input_height = 224

        rgb_img = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
        lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
        l_channel = lab_img[:, :, 0]

        l_channel_resize = cv.resize(l_channel, (input_width, input_height))
        l_channel_resize -= 50

        Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))
        ab_channel = Caffe_net.forward()[0, :, :, :].transpose((1, 2, 0))

        (original_height, original_width) = rgb_img.shape[:2]
        ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
        lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_channel_us), axis=2)
        bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)

        result_path = "./result.png"
        cv.imwrite(result_path, (bgr_output * 255).astype(np.uint8))

        load = Image.open(result_path)
        load = load.resize((480, 360), Image.ANTIALIAS)
        self.render2 = ImageTk.PhotoImage(load)

        if self.image is None:
            self.image = self.canvas.create_image(240, 180, image=self.render2)
        else:
            self.canvas.itemconfig(self.image, image=self.render2)

root = tk.Tk()
root.geometry("980x600")
root.title("B&W Image Colorization GUI")

app = Window(root)
app.pack(fill=tk.BOTH, expand=1)
root.mainloop()


      
