import sys
import os
import numpy as np
import random
import time

fractionOfDataUsedToTrain = .3
L1SIZE = 20
L2SIZE = 30
L3SIZE = 15
L4SIZE = 10
eta = .05
dataPointsPerBatch = 100

try:
    totalDataPointsAvailable = 0
    path = os.environ['MARKETDATADIR']
    Market_Data = os.listdir(path)
    for file in Market_Data:
        fullPath = os.path.join(path, file)
        with open(fullPath, 'r') as f:
            numDays = len(f.readlines())
            if numDays > L1SIZE:
                totalDataPointsAvailable += numDays - L1SIZE
except KeyError:
    raise KeyError('Environment variable "MARKETDATADIR" not set! Please set "MARKETDATADIR" to point where all market data should live first by appropriately updating variable in .bash_profile')

fractionOfTotalDataToUse = .1
#numTrainingEpochs = 1
numTrainingEpochs = int(((totalDataPointsAvailable / dataPointsPerBatch * fractionOfDataUsedToTrain) // 1) * fractionOfTotalDataToUse)
numTestingPoints = int(((totalDataPointsAvailable * (1 - fractionOfDataUsedToTrain)) // 1) * fractionOfTotalDataToUse)
steepnessOfCostFunction = 1

class net:

    def __init__(self, LAYER1SIZE, LAYER2SIZE, LAYER3SIZE, LAYER4SIZE, eta, dataPointsPerBatch, numTrainingEpochs, numTestingPoints):
        self.neuronizingFunction = self.SELU
        self.dNeuronizingFunctiondV = self.dSELUdV
        print("Using %r neuronizing function" %(self.neuronizingFunction.__name__))
        
        self.printCounter = 0
        
        self.LAYER1SIZE = LAYER1SIZE
        self.LAYER2SIZE = LAYER2SIZE
        self.LAYER3SIZE = LAYER3SIZE
        self.LAYER4SIZE = LAYER4SIZE
        self.LAYER5SIZE = 2
        self.eta = eta
        self.dataPointsPerBatch = dataPointsPerBatch
        self.numTrainingEpochs = numTrainingEpochs
        self.numTestingPoints = numTestingPoints
        self.dataObj = data(self.LAYER1SIZE)
        if self.neuronizingFunction == self.SELU:
            self.initializeWeightsLeCun()
        else:
            self.initializeWeightsXavier()

    def initializeWeightsXavier(self):
        print("\nUsing Xavier initializing function\n")
        self.layer2Biases = [0 for x in range (self.LAYER2SIZE)]
        self.layer3Biases = [0 for x in range (self.LAYER3SIZE)]
        self.layer4Biases = [0 for x in range (self.LAYER4SIZE)]
        self.layer5Biases = [0 for x in range (self.LAYER5SIZE)]
        
        constant = np.sqrt(6.) / np.sqrt(self.LAYER1SIZE + self.LAYER2SIZE)
        self.layer21Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER1SIZE)] \
            for y in range (self.LAYER2SIZE)]
        constant = np.sqrt(6.) / np.sqrt(self.LAYER2SIZE + self.LAYER3SIZE)
        self.layer32Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER2SIZE)] \
            for y in range (self.LAYER3SIZE)]
        constant = np.sqrt(6.) / np.sqrt(self.LAYER3SIZE + self.LAYER4SIZE)
        self.layer43Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER3SIZE)] \
            for y in range (self.LAYER4SIZE)]
        constant = np.sqrt(6.) / np.sqrt(self.LAYER4SIZE + self.LAYER5SIZE)
        self.layer54Weights = [[random.uniform(-1. * constant, constant) for x in range (self.LAYER4SIZE)] \
            for y in range (self.LAYER5SIZE)]
    
    def initializeWeightsLeCun(self):
        print("\nUsing LeCun initializing function\n")
        self.layer2Biases = [0 for x in range (self.LAYER2SIZE)]
        self.layer3Biases = [0 for x in range (self.LAYER3SIZE)]
        self.layer4Biases = [0 for x in range (self.LAYER4SIZE)]
        self.layer5Biases = [0 for x in range (self.LAYER5SIZE)]
    
        self.layer21Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER2SIZE)) for x in range (self.LAYER1SIZE)] \
            for y in range (self.LAYER2SIZE)]
        self.layer32Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER3SIZE)) for x in range (self.LAYER2SIZE)] \
            for y in range (self.LAYER3SIZE)]
        self.layer43Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER4SIZE)) for x in range (self.LAYER3SIZE)] \
            for y in range (self.LAYER4SIZE)]
        self.layer54Weights = [[np.random.normal(0, 1 / np.sqrt(self.LAYER5SIZE)) for x in range (self.LAYER4SIZE)] \
            for y in range (self.LAYER5SIZE)]

    def sendThroughNetTrain(self, inputData, trueResult):
        #Calculates output of neural net with input "inputData"
        if len(inputData) != self.LAYER1SIZE:
            raise ValueError("Input data is not of length %r" % self.LAYER1SIZE)
        layer1Values = inputData
        layer2Values = [0 for x in range (self.LAYER2SIZE)]
        layer3Values = [0 for x in range (self.LAYER3SIZE)]
        layer4Values = [0 for x in range (self.LAYER4SIZE)]
        layer5Values = [0 for x in range (self.LAYER5SIZE)]
        for L2Neuron in range (self.LAYER2SIZE):
            layer2Values[L2Neuron] = self.neuronizingFunction(self.layer2Biases[L2Neuron] + np.dot(layer1Values, self.layer21Weights[L2Neuron]))
        for L3Neuron in range (self.LAYER3SIZE):
            layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        for L4Neuron in range (self.LAYER4SIZE):
            layer4Values[L4Neuron] = self.neuronizingFunction(self.layer4Biases[L4Neuron] + np.dot(layer3Values, self.layer43Weights[L4Neuron]))
        for L5Neuron in range (self.LAYER5SIZE):
            layer5Values[L5Neuron] = self.layer5Biases[L5Neuron] + np.dot(layer4Values, self.layer54Weights[L5Neuron])
        layer5Values = self.softmax(layer5Values)
        if self.printCounter ==  0:
            print((layer5Values, trueResult))
        self.printCounter = (self.printCounter + 1) % 100
        squaredError = self.calculateSquaredError(layer5Values, trueResult)
        correctDirection = self.sameSign(self.directionize(layer5Values), trueResult)
        
        #Calculates gradient for training purposes
        gradientLayer21Weights = [[0 for x in range(self.LAYER1SIZE)] for y in range (self.LAYER2SIZE)]
        gradientLayer32Weights = [[0 for x in range(self.LAYER2SIZE)] for y in range (self.LAYER3SIZE)]
        gradientLayer43Weights = [[0 for x in range(self.LAYER3SIZE)] for y in range (self.LAYER4SIZE)]
        gradientLayer54Weights = [[0 for x in range(self.LAYER4SIZE)] for y in range (self.LAYER5SIZE)]
        
        gradientLayer2Biases = [0 for x in range(self.LAYER2SIZE)]
        gradientLayer3Biases = [0 for x in range(self.LAYER3SIZE)]
        gradientLayer4Biases = [0 for x in range(self.LAYER4SIZE)]
        gradientLayer5Biases = [0 for x in range(self.LAYER5SIZE)]
        
        for L5Neuron in range (self.LAYER5SIZE):
            flattened = (1. / (1. + np.exp(-1. * steepnessOfCostFunction * trueResult)))
            if (L5Neuron == 0):
                dCostdL5PostNeuronizingFunction = 2 * (layer5Values[L5Neuron] - flattened)
            else:
                dCostdL5PostNeuronizingFunction = 2 * (layer5Values[L5Neuron] - (1 - flattened))
            dL5PostNeuronizingFunctiondL5V = self.dSoftmaxdV(layer5Values, L5Neuron)
            gradientLayer5Biases[L5Neuron] = dCostdL5PostNeuronizingFunction * dL5PostNeuronizingFunctiondL5V
            
            for L4Neuron in range (self.LAYER4SIZE):
                dL5VdL54Weight = layer4Values[L4Neuron]
                gradientLayer54Weights[L5Neuron][L4Neuron] = gradientLayer5Biases[L5Neuron] * dL5VdL54Weight
            
                dL5VdL4PostNeuronizingFunction = self.layer54Weights[L5Neuron][L4Neuron]
                dL4PostNeuronizingFunctiondL4V = self.dNeuronizingFunctiondV(layer4Values[L4Neuron])
                gradientLayer4Biases[L4Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V)
        
                for L3Neuron in range (self.LAYER3SIZE):
                    dL4VdL43Weight = layer3Values[L3Neuron]
                    gradientLayer43Weights[L4Neuron][L3Neuron] += gradientLayer4Biases[L4Neuron] * dL4VdL43Weight
        
                    dL4VdL3PostNeuronizingFunction = self.layer43Weights[L4Neuron][L3Neuron]
                    dL3PostNeuronizingFunctiondL3V = self.dNeuronizingFunctiondV(layer3Values[L3Neuron])
                    gradientLayer3Biases[L3Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V)
            
                    for L2Neuron in range (self.LAYER2SIZE):
                        dL3VdL32Weight = layer2Values[L2Neuron]
                        gradientLayer32Weights[L3Neuron][L2Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V * dL3VdL32Weight)
                
                        dL3VdL2PostNeuronizingFunction = self.layer32Weights[L3Neuron][L2Neuron]
                        dL2PostNeuronizingFunctiondL2V = self.dNeuronizingFunctiondV(layer2Values[L2Neuron])
                        gradientLayer2Biases[L2Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V * dL3VdL2PostNeuronizingFunction * dL2PostNeuronizingFunctiondL2V)
                
                        for L1Neuron in range (self.LAYER1SIZE):
                            dL2VdL21Weights = layer1Values[L1Neuron]
                            gradientLayer21Weights[L2Neuron][L1Neuron] += (gradientLayer5Biases[L5Neuron] * dL5VdL4PostNeuronizingFunction * dL4PostNeuronizingFunctiondL4V * dL4VdL3PostNeuronizingFunction * dL3PostNeuronizingFunctiondL3V * dL3VdL2PostNeuronizingFunction * dL2PostNeuronizingFunctiondL2V * dL2VdL21Weights)
                    
        return (correctDirection, squaredError, gradientLayer21Weights, gradientLayer32Weights, gradientLayer43Weights, gradientLayer54Weights,
            gradientLayer2Biases, gradientLayer3Biases, gradientLayer4Biases, gradientLayer5Biases)

    def runBatch(self):
        #Runs a batch of data, logs average gradients and error
        totalCorrectDirection = 0
        totalSquaredError = 0
        totalGradientLayer21Weights = [[0 for x in range(self.LAYER1SIZE)] for y in range (self.LAYER2SIZE)]
        totalGradientLayer32Weights = [[0 for x in range(self.LAYER2SIZE)] for y in range (self.LAYER3SIZE)]
        totalGradientLayer43Weights = [[0 for x in range(self.LAYER3SIZE)] for y in range (self.LAYER4SIZE)]
        totalGradientLayer54Weights = [[0 for x in range(self.LAYER4SIZE)] for y in range (self.LAYER5SIZE)]
        totalGradientLayer2Biases = [0 for x in range(self.LAYER2SIZE)]
        totalGradientLayer3Biases = [0 for x in range(self.LAYER3SIZE)]
        totalGradientLayer4Biases = [0 for x in range(self.LAYER4SIZE)]
        totalGradientLayer5Biases = [0 for x in range(self.LAYER5SIZE)]
        
        for x in range (self.dataPointsPerBatch):
            inputData, trueResult = self.dataObj.getNewDataPoint()
            (correctDirection, newSquaredError, newGradientLayer21Weights, newGradientLayer32Weights, newGradientLayer43Weights, newGradientLayer54Weights, newGradientLayer2Biases, newGradientLayer3Biases, newGradientLayer4Biases, newGradientLayer5Biases) = self.sendThroughNetTrain(inputData, trueResult)
            if (correctDirection):
                totalCorrectDirection += 1
            totalSquaredError += newSquaredError
            totalGradientLayer21Weights = np.add(totalGradientLayer21Weights, newGradientLayer21Weights)
            totalGradientLayer32Weights = np.add(totalGradientLayer32Weights, newGradientLayer32Weights)
            totalGradientLayer43Weights = np.add(totalGradientLayer43Weights, newGradientLayer43Weights)
            totalGradientLayer54Weights = np.add(totalGradientLayer54Weights, newGradientLayer54Weights)
            totalGradientLayer2Biases = np.add(totalGradientLayer2Biases, newGradientLayer2Biases)
            totalGradientLayer3Biases = np.add(totalGradientLayer3Biases, newGradientLayer3Biases)
            totalGradientLayer4Biases = np.add(totalGradientLayer4Biases, newGradientLayer4Biases)
            totalGradientLayer5Biases = np.add(totalGradientLayer5Biases, newGradientLayer5Biases)

        correctDirectionRate = float(totalCorrectDirection) / float(self.dataPointsPerBatch)
        averageSquaredError = totalSquaredError / float(self.dataPointsPerBatch)
        averageGradientLayer21Weights = np.divide(totalGradientLayer21Weights, float(self.dataPointsPerBatch))
        averageGradientLayer32Weights = np.divide(totalGradientLayer32Weights, float(self.dataPointsPerBatch))
        averageGradientLayer43Weights = np.divide(totalGradientLayer43Weights, float(self.dataPointsPerBatch))
        averageGradientLayer54Weights = np.divide(totalGradientLayer54Weights, float(self.dataPointsPerBatch))
        averageGradientLayer2Biases = np.divide(totalGradientLayer2Biases, float(self.dataPointsPerBatch))
        averageGradientLayer3Biases = np.divide(totalGradientLayer3Biases, float(self.dataPointsPerBatch))
        averageGradientLayer4Biases = np.divide(totalGradientLayer4Biases, float(self.dataPointsPerBatch))
        averageGradientLayer5Biases = np.divide(totalGradientLayer5Biases, float(self.dataPointsPerBatch))
        
        #Updates weights and biases accordingly
        self.layer21Weights = np.subtract(self.layer21Weights, np.multiply(averageGradientLayer21Weights, self.eta))
        self.layer32Weights = np.subtract(self.layer32Weights, np.multiply(averageGradientLayer32Weights, self.eta))
        self.layer43Weights = np.subtract(self.layer43Weights, np.multiply(averageGradientLayer43Weights, self.eta))
        self.layer54Weights = np.subtract(self.layer54Weights, np.multiply(averageGradientLayer54Weights, self.eta))
        self.layer2Biases = np.subtract(self.layer2Biases, np.multiply(averageGradientLayer2Biases, self.eta))
        self.layer3Biases = np.subtract(self.layer3Biases, np.multiply(averageGradientLayer3Biases, self.eta))
        self.layer4Biases = np.subtract(self.layer4Biases, np.multiply(averageGradientLayer4Biases, self.eta))
        self.layer5Biases = np.subtract(self.layer5Biases, np.multiply(averageGradientLayer5Biases, self.eta))
        
        return averageSquaredError, correctDirectionRate
        
    def train(self):
        averageSquaredErrorProgress = correctDirectionRateProgress = 0
        for epoch in range (self.numTrainingEpochs):
            if (epoch == 1):
                start = time.perf_counter()
            averageSquaredError, correctDirectionRate  = self.runBatch()
            averageSquaredErrorProgress += averageSquaredError
            correctDirectionRateProgress += correctDirectionRate
            if (epoch == 1):
                end = time.perf_counter()
                timeElapsed = end - start
                print("\n[At this rate of %r sec/epoch, it will take approximately %r minutes, or %r hours, to train the neural net]\n" %(round(timeElapsed, 4), round((timeElapsed * self.numTrainingEpochs) / 60., 2), round((timeElapsed * self.numTrainingEpochs) / 3600., 3)))
            print("Epoch %r || Mean Squared Error = %r" %(epoch, round(averageSquaredError, 4)))
            if epoch % 20 == 0 and epoch != 0:
                print("\nPROGRESS TRACKER: MSE Avg. = %r || Correct Direction Rate = %r\n" %(round(averageSquaredErrorProgress / float(epoch), 4), round(correctDirectionRateProgress / float(epoch), 4)))
        
    def sendThroughNetTest(self, inputData, trueResult):
        #Calculates output of neural net with input "inputData"
        if len(inputData) != self.LAYER1SIZE:
            raise ValueError("Input data is not of length %r" % self.LAYER1SIZE)
        layer1Values = inputData
        layer2Values = [0 for x in range (self.LAYER2SIZE)]
        layer3Values = [0 for x in range (self.LAYER3SIZE)]
        layer4Values = [0 for x in range (self.LAYER4SIZE)]
        layer5Values = [0 for x in range (self.LAYER5SIZE)]
        for L2Neuron in range (self.LAYER2SIZE):
            layer2Values[L2Neuron] = self.neuronizingFunction(self.layer2Biases[L2Neuron] + np.dot(layer1Values, self.layer21Weights[L2Neuron]))
        for L3Neuron in range (self.LAYER3SIZE):
            layer3Values[L3Neuron] = self.neuronizingFunction(self.layer3Biases[L3Neuron] + np.dot(layer2Values, self.layer32Weights[L3Neuron]))
        for L4Neuron in range (self.LAYER4SIZE):
            layer4Values[L4Neuron] = self.neuronizingFunction(self.layer4Biases[L4Neuron] + np.dot(layer3Values, self.layer43Weights[L4Neuron]))
        for L5Neuron in range (self.LAYER5SIZE):
            layer5Values[L5Neuron] = self.layer5Biases[L5Neuron] + np.dot(layer4Values, self.layer54Weights[L5Neuron])
        layer5Values = self.softmax(layer5Values)
        squaredError = self.calculateSquaredError(layer5Values, trueResult)
        return layer5Values
        
    def test(self):
        trueTotalUp = 0
        guessedTotalUp = 0
        totalCorrectDirection = 0
        correctPointSixPlus = 0
        totalPointSixPlus = 0
        correctPointSevenPlus = 0
        totalPointSevenPlus = 0
        correctPointEightPlus = 0
        totalPointEightPlus = 0
        correctPointNinePlus = 0
        totalPointNinePlus = 0
        for test in range (numTestingPoints):
            inputData, trueResult = self.dataObj.getNewDataPoint()
            guessedResult = self.sendThroughNetTest(inputData, trueResult)
            guessedUp = (1 == self.directionize(guessedResult))
            if trueResult >= 0:
                trueTotalUp += 1
            if guessedUp:
                guessedTotalUp += 1
            correctDirection = self.sameSign(self.directionize(guessedResult), trueResult)
            if guessedResult[0] <= .1 or guessedResult[0] >= .9:
                totalPointNinePlus += 1
                totalPointEightPlus += 1
                totalPointSevenPlus += 1
                totalPointSixPlus += 1
                if correctDirection:
                    correctPointNinePlus += 1
                    correctPointEightPlus += 1
                    correctPointSevenPlus += 1
                    correctPointSixPlus += 1
            elif guessedResult[0] <= .2 or guessedResult[0] >= .8:
                totalPointEightPlus += 1
                totalPointSevenPlus += 1
                totalPointSixPlus += 1
                if correctDirection:
                    correctPointEightPlus += 1
                    correctPointSevenPlus += 1
                    correctPointSixPlus += 1
            elif guessedResult[0] <= .3 or guessedResult[0] >= .7:
                totalPointSevenPlus += 1
                totalPointSixPlus += 1
                if correctDirection:
                    correctPointSevenPlus += 1
                    correctPointSixPlus += 1
            elif guessedResult[0] <= .4 or guessedResult[0] >= .6:
                totalPointSixPlus += 1
                if correctDirection:
                    correctPointSixPlus += 1
            if correctDirection:
                totalCorrectDirection += 1
            print("Test %r || True Value = %r || Correct : %r || Guessed %r%% Up, %r%% Down" %(test, trueResult, correctDirection, round(guessedResult[0] * 100., 4), round(guessedResult[1] * 100., 4)))
        print("Testing over")
        print("%r fraction of days were truly positive" %(float(trueTotalUp) / float(numTestingPoints)))
        print("%r fraction of days were guessed to be positive" %(float(guessedTotalUp) / float(numTestingPoints)))
        print("%r fraction of days had their directions correctly guessed" %(float(totalCorrectDirection) / float(numTestingPoints)))
        if totalPointSixPlus > 0:
            ratio = float(correctPointSixPlus) / float(totalPointSixPlus)
        else:
            ratio = 0.
        print("%r fraction of days with confidence over .6 had their directions correctly guessed, or %r / %r" %(ratio, correctPointSixPlus, totalPointSixPlus))
        if totalPointSevenPlus > 0:
            ratio = float(correctPointSevenPlus) / float(totalPointSevenPlus)
        else:
            ratio = 0.
        print("%r fraction of days with confidence over .7 had their directions correctly guessed, or %r / %r" %(ratio, correctPointSevenPlus, totalPointSevenPlus))
        if totalPointEightPlus > 0:
            ratio = float(correctPointEightPlus) / float(totalPointEightPlus)
        else:
            ratio = 0.
        print("%r fraction of days with confidence over .8 had their directions correctly guessed, or %r / %r" %(ratio, correctPointEightPlus, totalPointEightPlus))
        if totalPointNinePlus > 0:
            ratio = float(correctPointNinePlus) / float(totalPointNinePlus)
        else:
            ratio = 0.
        print("%r fraction of days with confidence over .9 had their directions correctly guessed, or %r / %r" %(ratio, correctPointNinePlus, totalPointNinePlus))

    @staticmethod
    def sameSign(x, y):
        return ((x >= 0 and y >= 0) or (x < 0 and y < 0))

    @staticmethod
    def directionize(ls):
        if (ls[0] >= ls[1]):
            return 1
        else:
            return -1

    @staticmethod
    def calculateSquaredError(guess, trueResult):
        flattened = (1. / (1. + np.exp(-1. * steepnessOfCostFunction * trueResult)))
        return (guess[0] - flattened)**2 + (guess[1] - (1 - flattened))**2

    @staticmethod
    def softmax(ls):
        sum = 0
        m = max(ls)
        for x in ls:
            sum += np.exp(x - m)
        return [np.exp(i - m) / sum for i in ls]
        
    @staticmethod
    def dSoftmaxdV(ls, i):
        sum = 0
        m = max(ls)
        for x in ls:
            sum += np.exp(x - m)
        s = np.exp(i - m) / sum
        return s * (1 - s)

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))
        
    @staticmethod
    def dSoftplusdV(x):
        return (1. / (1. + np.exp(-1. * x)))
    
    @staticmethod
    def SELU(x):
        alpha = 1.6732632423543772848170429916717
        gamma = 1.0507009873554804934193349852946
        if x > 0:
            return gamma * x
        else:
            return gamma * alpha * (np.exp(x) - 1.)
        
    @staticmethod
    def dSELUdV(x):
        alpha = 1.6732632423543772848170429916717
        gamma = 1.0507009873554804934193349852946
        if x > 0:
            return gamma
        else:
            return gamma * alpha * np.exp(x)
        
    @staticmethod
    def sigmoid(x):
        return (1. / (1. + np.exp(-1. * x)))
        
    @staticmethod
    def dSigmoiddV(x):
        return (1. / (1. + np.exp(-1. * x))) * (1. - (1. / (1. + np.exp(-1. * x))))
    
    
class data:
    def __init__(self, size):
        self.size = size
        try:
            self.path = os.environ['MARKETDATADIR']
        except KeyError:
            raise KeyError('Environment variable "MARKETDATADIR" not set! Please set "MARKETDATADIR" to point where all market data should live first by appropriately updating variable in .bash_profile')
        self.permutedDataFiles = np.random.permutation(os.listdir(self.path))
        self.dataFileAt = 0
        self.getNewDataFile()
        
    def getNewDataFile(self):
        if self.dataFileAt >= len(self.permutedDataFiles):
            raise IndexError("Ran out of data files")
        fullPath = os.path.join(self.path, self.permutedDataFiles[self.dataFileAt])
        with open(fullPath, 'r') as f:
            self.currentDataFileInfo = f.readlines()
        self.currentDataFileInfo = [float(line.rstrip('\%\n')) for line in self.currentDataFileInfo]
        print("\n---------------NEW FILE OPENED: %r, file number %r--------------\n" %(self.permutedDataFiles[self.dataFileAt], self.dataFileAt))
        if (len(self.currentDataFileInfo) < 101):
            return self.getNewDataFile()
        self.dataFileAt += 1
        self.indexAtWithinPermutedIndices = 0
        self.permutedIndices = np.random.permutation([x for x in range (L1SIZE + 1, len(self.currentDataFileInfo))])
        
    def getNewDataPoint(self):
        while (self.indexAtWithinPermutedIndices >= len(self.permutedIndices)) or (self.indexAtWithinPermutedIndices >= len(self.permutedIndices) * (fractionOfTotalDataToUse + .01)):
            self.getNewDataFile()
        index = self.permutedIndices[self.indexAtWithinPermutedIndices]
        toReturn = self.currentDataFileInfo[index : index - self.size : -1], self.currentDataFileInfo[index - self.size]
        self.indexAtWithinPermutedIndices += 1
        #print(index)
        #print(toReturn)
        return toReturn
    
print("\nTraining neural net with the following parameters")
print("Number of Total Data Points Available : %r" %(totalDataPointsAvailable))
print("Layer 1 Size : %r neurons" %(L1SIZE))
print("Layer 2 Size : %r neurons" %(L2SIZE))
print("Layer 3 Size : %r neurons" %(L3SIZE))
print("Layer 4 Size : %r neurons" %(L4SIZE))
print("Layer 5 Size : 2 neurons")
print("eta : %r" %(eta))
print("Data Points per Batch : %r" %(dataPointsPerBatch))
print("Number of Training Epochs : %r" %(numTrainingEpochs))
print("Number of Testing Points : %r" %(numTestingPoints))
print("Fraction of Total Data Used: %r" %(fractionOfTotalDataToUse))

network = net(L1SIZE, L2SIZE, L3SIZE, L4SIZE, eta, dataPointsPerBatch, numTrainingEpochs, numTestingPoints)
network.train()
print("\n------------END TRAINING------------")
print("------------BEGIN TESTING------------\n")
network.test()
print("\n------------END TESTING------------\n")

print("Neural net was trained with the following parameters")
print("Number of Total Data Points Available : %r" %(totalDataPointsAvailable))
print("Layer 1 Size : %r neurons" %(L1SIZE))
print("Layer 2 Size : %r neurons" %(L2SIZE))
print("Layer 3 Size : %r neurons" %(L3SIZE))
print("Layer 4 Size : %r neurons" %(L4SIZE))
print("Layer 5 Size : 2 neurons")
print("Steepness of Cost Function : %r" %(steepnessOfCostFunction))
print("eta : %r" %(eta))
print("Data Points per Batch : %r" %(dataPointsPerBatch))
print("Number of Training Epochs : %r" %(numTrainingEpochs))
print("Number of Testing Points : %r" %(numTestingPoints))
print("Fraction of Total Data Used: %r" %(fractionOfTotalDataToUse))
