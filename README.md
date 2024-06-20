
# B&W Image Colorization GUI

This project is a graphical user interface (GUI) application for colorizing black and white images using deep learning. The application is built with Python, utilizing the Tkinter library for the GUI, and OpenCV for image processing and colorization.

## Features

- Upload black and white images.
- Colorize uploaded images using a pre-trained deep learning model.
- Save the colorized images.

 
![result](https://github.com/madhu1690/Image-Colorization/assets/135344672/6d32538d-36d2-44d9-aee6-8f16cfe3b4a9)







## Requirements
- Python 3.x
- Tkinter
- OpenCV
- NumPy
- PIL (Pillow)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/colorization-gui.git
    cd colorization-gui
    ```

2. **Install the required packages:**
    ```bash
    pip install numpy opencv-python pillow
    ```

3. **Download the required model files:**
    - [colorization_deploy_v2.prototxt](https://github.com/richzhang/colorization/blob/master/models/colorization_deploy_v2.prototxt)
    - [colorization_release_v2.caffemodel](https://github.com/richzhang/colorization/blob/master/models/colorization_release_v2.caffemodel)
    - [pts_in_hull.npy](https://github.com/richzhang/colorization/blob/master/resources/pts_in_hull.npy)



## Usage

1. **Run the application:**
    ```bash
    python main.py
    ```

2. **Upload an image:**
    - Click on "File" -> "Upload Image".
    - Select a black and white image from your computer.

3. **Colorize the image:**
    - Click on "File" -> "Color Image".
    - The colorized image will be displayed on the canvas.

## Code Explanation

The main components of the code are:

- **Window Class**: Defines the GUI window and its elements.
    - `uploadImage()`: Method to upload an image and display it on the canvas.
    - `color()`: Method to colorize the uploaded image and display the result on the canvas.

- **Model Loading and Image Processing**: 
    - The pre-trained Caffe model is loaded using OpenCV's DNN module.
    - The black and white image is processed, resized, and fed to the model for colorization.
    - The output is then converted back to an image and displayed.

## File Structure

Certainly! Here is a README.md file for your B&W Image Colorization project using Tkinter and OpenCV:

markdown
Copy code
# B&W Image Colorization GUI

This project is a graphical user interface (GUI) application for colorizing black and white images using deep learning. The application is built with Python, utilizing the Tkinter library for the GUI, and OpenCV for image processing and colorization.

## Features

- Upload black and white images.
- Colorize uploaded images using a pre-trained deep learning model.
- Save the colorized images.

## Requirements

- Python 3.x
- Tkinter
- OpenCV
- NumPy
- PIL (Pillow)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/colorization-gui.git
    cd colorization-gui
    ```

2. **Install the required packages:**
    ```bash
    pip install numpy opencv-python pillow
    ```

3. **Download the required model files:**
    - [colorization_deploy_v2.prototxt](https://github.com/richzhang/colorization/blob/master/models/colorization_deploy_v2.prototxt)
    - [colorization_release_v2.caffemodel](https://github.com/richzhang/colorization/blob/master/models/colorization_release_v2.caffemodel)
    - [pts_in_hull.npy](https://github.com/richzhang/colorization/blob/master/resources/pts_in_hull.npy)

   Place these files in the `models` directory or update the paths in the script.

## Usage

1. **Run the application:**
    ```bash
    python main.py
    ```

2. **Upload an image:**
    - Click on "File" -> "Upload Image".
    - Select a black and white image from your computer.

3. **Colorize the image:**
    - Click on "File" -> "Color Image".
    - The colorized image will be displayed on the canvas.

## Code Explanation

The main components of the code are:

- **Window Class**: Defines the GUI window and its elements.
    - `uploadImage()`: Method to upload an image and display it on the canvas.
    - `color()`: Method to colorize the uploaded image and display the result on the canvas.

- **Model Loading and Image Processing**: 
    - The pre-trained Caffe model is loaded using OpenCV's DNN module.
    - The black and white image is processed, resized, and fed to the model for colorization.
    - The output is then converted back to an image and displayed.

## File Structure

colorization-gui/
│
├── models/
│ ├── colorization_deploy_v2.prototxt
│ ├── colorization_release_v2.caffemodel
│ └── pts_in_hull.npy
│
├── main.py
└── README.md


