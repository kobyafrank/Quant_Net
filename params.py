import sys
import os

sizeOfInput = 50
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
    
layerSizeTuple = (sizeOfInput, 10)
neuronizingFunction = "Sigmoid"
eta = .2
dataPointsPerBatch = 150
fractionOfTotalDataToUse = .3
fractionOfDataUsedToTrain = .5
numTrainingEpochs = 5
#numTrainingEpochs = int(((totalDataPointsAvailable / dataPointsPerBatch * fractionOfDataUsedToTrain) // 1) * fractionOfTotalDataToUse)
numTestingPoints = int(((totalDataPointsAvailable * (1 - fractionOfDataUsedToTrain)) // 1) * fractionOfTotalDataToUse)
steepnessOfCostFunction = .7

typeOfNet = "Feed Forward"
