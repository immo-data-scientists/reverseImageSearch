import pathlib
from numpy import *
import cv2


def twoD2OneD(trainingImagePath):
    trainingImageRoot = pathlib.Path(trainingImagePath)
    images = list(trainingImageRoot.glob('*'))
    images = [str(path) for path in images]
    imageCount = len(images)
    faceMatrix = mat(zeros((imageCount,250*250)))
    j=0
    t=0
    for iImg in range(imageCount):
        imagePath = images[:iImg+1][0]
        img = cv2.imread(imagePath,0)
        faceMatrix[t,:] = mat(img).flatten()
        t=t+1                      
    return faceMatrix