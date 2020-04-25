import sys
import os
import numpy as np
import random

class net:
    
    def __init__(self, LAYER1SIZE, LAYER2SIZE, LAYER3SIZE, eta, dataPointsPerBatch, numEpochs):
        self.maxInitialWeight = .2
        self.LAYER1SIZE = LAYER1SIZE
        self.LAYER2SIZE = LAYER2SIZE
        self.LAYER3SIZE = LAYER3SIZE
        self.eta = eta
        self.dataPointsPerBatch = dataPointsPerBatch
        self.numEpochs = numEpochs
        self.initializeWeights(self.maxInitialWeight)
        self.dataObj = data(self.LAYER1SIZE)

    def initializeWeights(self, maxInitialWeight):
        #Initializes biases and weights for random small floats
        self.layer2Biases = [random.uniform(-1 * maxInitialWeight, maxInitialWeight) for x in range (self.LAYER2SIZE)]
        self.layer3Biases = [random.uniform(-1 * maxInitialWeight, maxInitialWeight) for x in range (self.LAYER3SIZE)]
        self.layer4Bias = random.uniform(-1 * maxInitialWeight, maxInitialWeight)
        
        self.layer21Weights = [[random.uniform(-1 * maxInitialWeight, maxInitialWeight) for x in range (self.LAYER1SIZE)] \
            for y in range (self.LAYER2SIZE)]
        self.layer32Weights = [[random.uniform(-1 * maxInitialWeight, maxInitialWeight) for x in range (self.LAYER2SIZE)] \
            for y in range (self.LAYER3SIZE)]
        self.layer43Weights = [random.uniform(-1 * maxInitialWeight, maxInitialWeight) for x in range (self.LAYER3SIZE)]

    def sendThroughNetTrain(self, inputData, trueResult):
        #Calculates output of neural net with input "inputData"
        if len(inputData) != self.LAYER1SIZE:
            raise ValueError("Input data is not of length %r" % self.LAYER1SIZE)
        layer1Values = [self.neuronizeMarketData(x) for x in inputData]
        layer2Values = [0 for x in range (self.LAYER2SIZE)]
        layer3Values = [0 for x in range (self.LAYER3SIZE)]
        layer4Value = 0
        for L2Neuron in range (self.LAYER2SIZE):
            layer2Values[L2Neuron] = self.sigmoid(self.layer2Biases[L2Neuron] + np.dot(layer1Values, self.layer21Weights[L2Neuron]))
        for L3Neuron in range (self.LAYER3SIZE):
            layer3Values[L3Neuron] = self.sigmoid(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        layer4Value = self.sigmoid(self.layer4Bias + np.dot(layer3Values, self.layer43Weights))
        squaredError = self.calculateSquaredError(layer4Value, self.neuronizeMarketData(trueResult))
        correctDirection = self.sameSign(trueResult >= 0, self.inverseNeuronizeMarketData(layer4Value))
        
        #Calculates gradient for training purposes
        gradientLayer21Weights = [[0 for x in range(self.LAYER1SIZE)] for y in range (self.LAYER2SIZE)]
        gradientLayer32Weights = [[0 for x in range(self.LAYER2SIZE)] for y in range (self.LAYER3SIZE)]
        gradientLayer43Weights = [0 for x in range(self.LAYER3SIZE)]
        
        gradientLayer2Biases = [0 for x in range(self.LAYER2SIZE)]
        gradientLayer3Biases = [0 for x in range(self.LAYER3SIZE)]
        gradientLayer4Bias = 0

        dCostdL4PostSigmoid = 2 * (layer4Value - trueResult)
        dL4PostSigmoiddL4PreSigmoid = self.sigmoidPrime(layer4Value)
        gradientLayer4Bias = dCostdL4PostSigmoid * dL4PostSigmoiddL4PreSigmoid
        
        for L3Neuron in range (self.LAYER3SIZE):
            dL4PreSigmoiddL43Weight = layer3Values[L3Neuron]
            gradientLayer43Weights[L3Neuron] = gradientLayer4Bias * dL4PreSigmoiddL43Weight
        
            dL4PreSigmoiddL3PostSigmoid = self.layer43Weights[L3Neuron]
            dL3PostSigmoiddL3PreSigmoid = self.sigmoidPrime(layer3Values[L3Neuron])
            gradientLayer3Biases[L3Neuron] = gradientLayer4Bias * dL4PreSigmoiddL3PostSigmoid * dL3PostSigmoiddL3PreSigmoid
            
            for L2Neuron in range (self.LAYER2SIZE):
                dL3PreSigmoiddL32Weight = layer2Values[L2Neuron]
                gradientLayer32Weights[L3Neuron][L2Neuron] = gradientLayer3Biases[L3Neuron] * dL3PreSigmoiddL32Weight
                
                dL3PreSigmoiddL2PostSigmoid = self.layer32Weights[L3Neuron][L2Neuron]
                dL2PostSigmoiddL2PreSigmoid = self.sigmoidPrime(layer2Values[L2Neuron])
                gradientLayer2Biases[L2Neuron] = (gradientLayer3Biases[L3Neuron] * dL3PreSigmoiddL2PostSigmoid * dL2PostSigmoiddL2PreSigmoid)
                
                for L1Neuron in range (self.LAYER1SIZE):
                    dL2PreSigmoiddL21Weights = layer1Values[L1Neuron]
                    gradientLayer21Weights[L2Neuron][L1Neuron] = gradientLayer2Biases[L2Neuron] * dL2PreSigmoiddL21Weights
                    
        return (correctDirection, squaredError, gradientLayer21Weights, gradientLayer32Weights, gradientLayer43Weights,
            gradientLayer2Biases, gradientLayer3Biases, gradientLayer4Bias)

    def sendThroughNetTest(self, inputData, trueResult):
        #Calculates output of neural net with input "inputData"
        if len(inputData) != self.LAYER1SIZE:
            raise ValueError("Input data is not of length %r" % self.LAYER1SIZE)
        layer1Values = [self.neuronizeMarketData(x) for x in inputData]
        layer2Values = [0 for x in range (self.LAYER2SIZE)]
        layer3Values = [0 for x in range (self.LAYER3SIZE)]
        layer4Value = 0
        for L2Neuron in range (self.LAYER2SIZE):
            layer2Values[L2Neuron] = self.sigmoid(self.layer2Biases[L2Neuron] + np.dot(layer1Values, self.layer21Weights[L2Neuron]))
        for L3Neuron in range (self.LAYER3SIZE):
            layer3Values[L3Neuron] = self.sigmoid(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        layer4Value = self.sigmoid(self.layer4Bias + np.dot(layer3Values, self.layer43Weights))
        return self.inverseNeuronizeMarketData(layer4Value), trueResult

    def runBatch(self):
        #Runs a batch of data, logs average gradients and error
        totalCorrectDirection = 0
        totalSquaredError = 0
        totalGradientLayer21Weights = [[0 for x in range(self.LAYER1SIZE)] for y in range (self.LAYER2SIZE)]
        totalGradientLayer32Weights = [[0 for x in range(self.LAYER2SIZE)] for y in range (self.LAYER3SIZE)]
        totalGradientLayer43Weights = [0 for x in range(self.LAYER3SIZE)]
        totalGradientLayer2Biases = [0 for x in range(self.LAYER2SIZE)]
        totalGradientLayer3Biases = [0 for x in range(self.LAYER3SIZE)]
        totalGradientLayer4Bias = 0
        
        for x in range (self.dataPointsPerBatch):
            (correctDirection, newSquaredError, newGradientLayer21Weights, newGradientLayer32Weights, newGradientLayer43Weights,newGradientLayer2Biases, newGradientLayer3Biases, newGradientLayer4Bias) = self.sendThroughNetTrain(dataObj.getNewDataPoint()))
            
            if (correctDirection):
                totalCorrectDirection += 1
            totalSquaredError += newSquaredError
            totalGradientLayer21Weights = np.add(totalGradientLayer21Weights, newGradientLayer21Weights)
            totalGradientLayer32Weights = np.add(totalGradientLayer32Weights, newGradientLayer32Weights)
            totalGradientLayer43Weights = np.add(totalGradientLayer43Weights, newGradientLayer43Weights)
            totalGradientLayer2Biases = np.add(totalGradientLayer2Biases, newGradientLayer2Biases)
            totalGradientLayer3Biases = np.add(totalGradientLayer3Biases, newGradientLayer3Biases)
            totalGradientLayer4Bias += newGradientLayer4Bias

        correctDirectionRate = totalCorrectDirection / float(self.dataPointsPerBatch)
        averageSquaredError = totalSquaredError / float(self.dataPointsPerBatch)
        averageGradientLayer21Weights = np.divide(totalGradientLayer21Weights, float(self.dataPointsPerBatch))
        averageGradientLayer32Weights = np.divide(totalGradientLayer32Weights, float(self.dataPointsPerBatch))
        averageGradientLayer43Weights = np.divide(totalGradientLayer43Weights, float(self.dataPointsPerBatch))
        averageGradientLayer2Biases = np.divide(totalGradientLayer2Biases, float(self.dataPointsPerBatch))
        averageGradientLayer3Biases = np.divide(totalGradientLayer3Biases, float(self.dataPointsPerBatch))
        averageGradientLayer4Bias = totalGradientLayer4Bias / float(self.dataPointsPerBatch)
        
        #Updates weights and biases accordingly
        self.layer21Weights = np.add(self.layer21Weights, np.multiply(averageGradientLayer21Weights, self.eta))
        self.layer32Weights = np.add(self.layer32Weights, np.multiply(averageGradientLayer32Weights, self.eta))
        self.layer43Weights = np.add(self.layer43Weights, np.multiply(averageGradientLayer43Weights, self.eta))
        self.layer2Biases = np.add(self.layer2Biases, np.multiply(averageGradientLayer2Biases, self.eta))
        self.layer3Biases = np.add(self.layer3Biases, np.multiply(averageGradientLayer3Biases, self.eta))
        self.layer4Bias = self.layer4Bias + self.eta * averageGradientLayer4Bias
        
        return averageSquaredError, correctDirectionRate
        
    def train(self):
        for epoch in range (self.numEpochs):
            averageSquaredError, correctDirectionRate  = self.runBatch()
            print("Epoch %r || Average Error = %r || Correct Direction Rate = %r" %(epoch, averageSquaredError, correctDirectionRate))
                
    def test(self):
        trueTotalUp = 0
        guessedTotalUp = 0
        totalCorrectDirection = 0
        totalAbsoluteError = 0
        for test in range (numTestingPoints):
            guessedResult, trueResult = self.sendThroughNetTest(dataObj.getNewDataPoint())
            if trueResult >= 0:
                trueTotalUp += 1
            if guessedTotalUp >= 0:
                guessedTotalUp += 1
            if self.sameSign(guessedResult, trueResult):
                totalCorrectDirection += 1
            totalAbsoluteError += np.abs(trueResult - guessedResult)
            print("Test %r || True Value = %r || Guessed Value = %r || Absolute Error = %r || Correct Direction : %r" %(test, trueResult, guessedResult, trueResult - guessedResult, self.sameSign(trueResult, guessedResult)))
        print("Testing over")
        print("%r\% of days were truly positive" %(trueTotalUp / numTestingPoints))
        print("%r\% of days were guessed to be positive" %(guessedTotalUp / numTestingPoints))
        print("%r\% of days had their directions correctly guessed" %(totalCorrectDirection / numTestingPoints))
        print("Average absolute error is %r \% per day" %(totalAbsoluteError / numTestingPoints))



    @staticmethod
    def sameSign(x, y):
        return ((x >= 0 and y >= 0) or (x < 0 and y < 0))

    @staticmethod
    def calculateSquaredError(guess, trueResult):
        return (guess - trueResult) * (guess - trueResult)

    @staticmethod
    def neuronizeMarketData(x):
    #TO CHANGE LATER MAYBE
        return (1. / (1. + np.exp(-1. * x)))

    @staticmethod
    def inverseNeuronizeMarketData(x):
    #TO CHANGE LATER MAYBE
        return np.log(x / (1. - x))
        
    @staticmethod
    def inverseSigmoid(x):
        return np.log(x / (1. - x))

    @staticmethod
    def sigmoid(x):
        return (1. / (1. + np.exp(-1. * x)))
        
    @staticmethod
    def sigmoidPrime(x):
        return (1. / (1. + np.exp(-1. * x))) * (1. - (1. / (1. + np.exp(-1. * x))))
    
    
class data:
    def __init__(self, size):
        self.size = size
        self.dataSet = [0 for x in range (self.size)]
        self.trueResult = 0
        self.indexAt = 0
        try:
            path = os.environ['MARKETDATADIR']
        except KeyError:
            raise KeyError('Environment variable "MARKETDATADIR" not set! Please set "MARKETDATADIR" to point where all market data should live first by appropriately updating variable in ./bash_profile')
        self.permutedDataFiles = np.random.permutation(os.listdir(self.path))
        
        
        
        
    def getNewDataPoint():

            with open(os.path.join(os.cwd(), filename), 'r') as f:


    return self.dataSet[indexAt:] + self.dataSet[:indexAt], selftrueResult
    
    
newNetwork = net(5, 200, 200, .2, 5, 10)
newNetwork.train()
