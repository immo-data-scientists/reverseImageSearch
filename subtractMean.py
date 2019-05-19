def subtractMean(trainingData,mean):    
    subtractedMean = []
    imageCount = len(trainingData)
    for iImg in range(imageCount):
        tempData = trainingData[:iImg+1] - mean
        # print('*****', iImg, temp)
        subtractedMean.append(tempData)
    return subtractedMean