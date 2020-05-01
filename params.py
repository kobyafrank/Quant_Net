import sys
import os

sizeOfInput = 30
try:
    totalDataPointsAvailable = 0
    path = os.environ['MARKETDATADIR']
    Market_Data = os.listdir(path)
    for file in Market_Data:
        fullPath = os.path.join(path, file)
        with open(fullPath, 'r') as f:
            numDays = len(f.readlines())
            if numDays > sizeOfInput:
                totalDataPointsAvailable += numDays - sizeOfInput
except KeyError:
    raise KeyError('Environment variable "MARKETDATADIR" not set! Please set "MARKETDATADIR" to point where all market data should live first by appropriately updating variable in .bash_profile')
    
layerSizeTuple = (sizeOfInput, 20, 10)
neuronizingFunction = "selu"
eta = .1
dropoutRate = 0.5
dataPointsPerBatch = 150
fractionOfTotalDataToUse = .5
fractionOfDataUsedToTrain = .5
#numTrainingEpochs = 10
numTrainingEpochs = int(((totalDataPointsAvailable / dataPointsPerBatch * fractionOfDataUsedToTrain) // 1) * fractionOfTotalDataToUse)
numTestingPoints = int(((totalDataPointsAvailable * (1 - fractionOfDataUsedToTrain)) // 1) * fractionOfTotalDataToUse)
steepnessOfCostFunction = 1.5

typeOfNet = "Feed Forward"
