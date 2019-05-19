#python imageProcessing/BirdsIdentificationBasedOnImages.py
from numpy import * 
import numpy as np
from numpy import linalg as la
import cv2
import os

from TwoD2OneD import twoD2OneD
from Recognition import recognition
from EigenFaces import eigenFaces

# BIRDS DATABASE
trainingImagePath = 'T:/TENSORFLOW/ImageRecognition/images/'

# IMAGE FOR TESTING
testImagePath = 'T:/TENSORFLOW/ImageRecognition/testImages/'
testImage = 'owl.jpg' #'Rose-Ringed-Parakeet.jpg'

# FIND EIGEN FACES FROM COVARIANCE MATRIX
meanImg,eigenfaces,difference = eigenFaces(threshHold = 0.9,trainingImagePath=trainingImagePath)

# RECOGNISE IMAGE AND RETURN THE INDEX OF IMAGE
imagePath = testImagePath + testImage
testImage = cv2.imread(imagePath,0)
testImage = cv2.resize(testImage,(250,250))
indexOfImage = recognition(mat(testImage).flatten(),eigenfaces,meanImg,difference)
print("Index Of Image",indexOfImage)

# MAP IMAGE TO DATASET
