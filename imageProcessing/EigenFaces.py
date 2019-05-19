# FIND EIGEN FACES FROM COVARIANCE MATRIX
from numpy import *
from TwoD2OneD import twoD2OneD

def eigenFaces(threshHold = 0.85,trainingImagePath=''):    
    FaceMat = twoD2OneD(trainingImagePath).T 
    avgImg = mean(FaceMat,1)
    diffTrain = FaceMat-avgImg
    eigvals,eigVects = linalg.eig(mat(diffTrain.T*diffTrain))
    eigSortIndex = argsort(-eigvals)
    for i in range(shape(FaceMat)[1]):
        if ((len(eigvals[eigSortIndex[:i]])==0 and eigvals[eigSortIndex[:i]].sum()>0) and (eigvals[eigSortIndex[:i]]/eigvals.sum()).sum() >= threshHold):
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:,eigSortIndex] 
    return avgImg,covVects,diffTrain