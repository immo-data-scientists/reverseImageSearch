#python BirdsIdentificationBasedOnImages.py
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pathlib
from os.path import join
from keras.preprocessing import image
from TwoD2OneD import twoD2oneD
from subtractMean import subtractMean
import numpy as np
import cv2

# BIRDS DATABASE
trainingImagePath = 'T:/TENSORFLOW/ImageRecognition/images/'

# IMAGE FOR TESTING
testImagePath = 'T:/TENSORFLOW/ImageRecognition/images/'
testImage = 'Rose-Ringed-Parakeet.jpg'
# img = image.load_img(join(testImagePath, testImage), target_size=(224, 224))
# img.show()

# CONVERT IMAGES FROM 2D TO 1D
trainingImageRoot = pathlib.Path(trainingImagePath)
images = list(trainingImageRoot.glob('*'))

images = [str(path) for path in images]
trainingData = twoD2oneD(images)

#FIND MEAN OF TRAINING IMAGES
meanData = np.mean(trainingData,axis=0)
subtractedMean = subtractMean(trainingData, meanData)

# np.cov(x)
# COMPUTE EIGEN VECTOR
mean, eigenVectors = cv2.PCACompute(trainingData, mean=None)


