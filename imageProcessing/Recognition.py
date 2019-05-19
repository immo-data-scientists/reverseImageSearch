# RECOGNISE IMAGE USING EUCLIDIAN DISTANCE AND RETURN THE INDEX OF IMAGE
from numpy import *

def recognition(testImage,eigenfaces,meanImg,difference):
    diff = testImage.T - meanImg
    eigenVector = eigenfaces.T* diff
    indexOfImage = 0
    euclidianDist = inf
    
    for i in range(3):
        TrainVec = eigenfaces.T*difference[:,i]
        if  (array(eigenVector-TrainVec)**2).sum() < euclidianDist:
            indexOfImage =  i
            euclidianDist = (array(eigenVector-TrainVec)**2).sum()
    return indexOfImage+1