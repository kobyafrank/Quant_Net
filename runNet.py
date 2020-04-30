import sys
import os
from feedForwardNet import feedForwardNet
from LSTMNet import LSTMNet

sizeOfInput = 50
inTuple = (sizeOfInput, 25, 10)
neuronizingFunction = "SELU"
eta = .1
dataPointsPerBatch = 100
steepnessOfCostFunction = .7

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

fractionOfDataUsedToTrain = .5
fractionOfTotalDataToUse = .3
numTrainingEpochs = 5
#numTrainingEpochs = int(((totalDataPointsAvailable / dataPointsPerBatch * fractionOfDataUsedToTrain) // 1) * fractionOfTotalDataToUse)
numTestingPoints = int(((totalDataPointsAvailable * (1 - fractionOfDataUsedToTrain)) // 1) * fractionOfTotalDataToUse)

network = feedForwardNet(inTuple, neuronizingFunction, eta, dataPointsPerBatch, numTrainingEpochs, numTestingPoints, fractionOfTotalDataToUse, steepnessOfCostFunction)

print("\nTraining neural net with the following parameters")
print("Number of Total Data Points Available : %r" %(totalDataPointsAvailable))
print("Fraction of Total Data Used: %r" %(fractionOfTotalDataToUse))
print("Steepness of Cost Function = %r" %(network.steepnessOfCostFunction))
print("Number of Layers : %r layers" %(network.numLayers))
print("Layer 1 Size : %r neurons" %(network.LAYER1SIZE))
print("Layer 2 Size : %r neurons" %(network.LAYER2SIZE))
print("Layer 3 Size : %r neurons" %(network.LAYER3SIZE))
print("Layer 4 Size : %r neurons" %(network.LAYER4SIZE))
print("Layer 5 Size : %r neurons" %(network.LAYER5SIZE))
print("Using %r neuronizing function" %(network.neuronizingFunction.__name__))
print("eta : %r" %(network.eta))
print("Data Points per Batch : %r" %(network.dataPointsPerBatch))
print("Number of Training Epochs : %r" %(network.numTrainingEpochs))
print("Number of Testing Points : %r" %(network.numTestingPoints))

network.train()
print("\n------------END TRAINING------------\n")
print("\n------------BEGIN TESTING------------\n")
network.test()
print("\n------------END TESTING------------\n")

print("Neural net was trained with the following parameters")
print("Number of Total Data Points Available : %r" %(totalDataPointsAvailable))
print("Fraction of Total Data Used: %r" %(fractionOfTotalDataToUse))
print("Steepness of Cost Function = %r" %(network.steepnessOfCostFunction))
print("Number of Layers : %r layers" %(network.numLayers))
print("Layer 1 Size : %r neurons" %(network.LAYER1SIZE))
print("Layer 2 Size : %r neurons" %(network.LAYER2SIZE))
print("Layer 3 Size : %r neurons" %(network.LAYER3SIZE))
print("Layer 4 Size : %r neurons" %(network.LAYER4SIZE))
print("Layer 5 Size : %r neurons" %(network.LAYER5SIZE))
print("Using %r neuronizing function" %(network.neuronizingFunction.__name__))
print("eta : %r" %(network.eta))
print("Data Points per Batch : %r" %(network.dataPointsPerBatch))
print("Number of Training Epochs : %r" %(network.numTrainingEpochs))
print("Number of Testing Points : %r" %(network.numTestingPoints))
