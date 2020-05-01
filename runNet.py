from feedForwardNet import feedForwardNet
import params

if params.typeOfNet in ["FEED FORWARD", "Feed Forward", "feed forward", "feedForward", "FF", "ff"]:
    network = feedForwardNet()
elif params.typeOfNet in ["LSTM", "Lstm", "lstm"]:
    network = LSTMNet()
else:
    raise ValueError("Invalid type of neural net. Please choose between Feed Forward and LSTM")

print("\nTraining neural net with the following parameters")
print("Number of Total Data Points Available : %r" %(params.totalDataPointsAvailable))
print("Fraction of Total Data Used: %r" %(params.fractionOfTotalDataToUse))
print("Steepness of Cost Function = %r" %(network.steepnessOfCostFunction))
print("Number of Layers : %r layers" %(network.numLayers))
print("Layer 1 Size : %r neurons" %(network.LAYER1SIZE))
print("Layer 2 Size : %r neurons" %(network.LAYER2SIZE))
print("Layer 3 Size : %r neurons" %(network.LAYER3SIZE))
print("Layer 4 Size : %r neurons" %(network.LAYER4SIZE))
print("Layer 5 Size : %r neurons" %(network.LAYER5SIZE))
print("Using %r neuronizing function" %(network.neuronizingFunction.__name__))
print("eta : %r" %(network.eta))
print("Dropout Rate : %r" %(network.dropoutRate))
print("Data Points per Batch : %r" %(network.dataPointsPerBatch))
print("Number of Training Epochs : %r" %(network.numTrainingEpochs))
print("Number of Testing Points : %r" %(network.numTestingPoints))

network.train()
print("\n------------END TRAINING------------\n")
print("\n------------BEGIN TESTING------------\n")
network.test()
print("\n------------END TESTING------------\n")

print("Neural net was trained with the following parameters")
print("Number of Total Data Points Available : %r" %(params.totalDataPointsAvailable))
print("Fraction of Total Data Used: %r" %(params.fractionOfTotalDataToUse))
print("Steepness of Cost Function = %r" %(network.steepnessOfCostFunction))
print("Number of Layers : %r layers" %(network.numLayers))
print("Layer 1 Size : %r neurons" %(network.LAYER1SIZE))
print("Layer 2 Size : %r neurons" %(network.LAYER2SIZE))
print("Layer 3 Size : %r neurons" %(network.LAYER3SIZE))
print("Layer 4 Size : %r neurons" %(network.LAYER4SIZE))
print("Layer 5 Size : %r neurons" %(network.LAYER5SIZE))
print("Used %r neuronizing function" %(network.neuronizingFunction.__name__))
print("eta : %r" %(network.eta))
print("Dropout Rate : %r" %(network.dropoutRate))
print("Data Points per Batch : %r" %(network.dataPointsPerBatch))
print("Number of Training Epochs : %r" %(network.numTrainingEpochs))
print("Number of Testing Points : %r" %(network.numTestingPoints))
