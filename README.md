# facedetection-opencv
## - About
Face Detection using Viola-Jones algorithm  to identify Haar-like features in a frame, taken from the webcam of the device.

## - Packages Used

Python wrapper for openCV:

    pip install opencv-python
This package install both openCV and numpy.

It also contains basic cascades for Haar-like features such as faces,smile, eyes, etc.
They can be accesed by:

    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
