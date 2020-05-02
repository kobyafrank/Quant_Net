import sys
import os
import numpy as np
import random

class data:
    def __init__(self, size, fractionOfTotalDataToUse, type = "r"):
        if type in ["rand", "random", "Rand", "Random", "r", "R"]:
            self.type = "r"
        elif type in ["Sliding", "sliding", "S", "s"]:
            self.type = "s"
        else:
            raise ValueError("Invalid type. Choose between Random and Sliding")
        self.size = size
        self.fractionOfTotalDataToUse = fractionOfTotalDataToUse
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
        if self.type == "r":
            self.permutedIndices = np.random.permutation([x for x in range (self.size + 1, len(self.currentDataFileInfo))])
        elif self.type == "s":
            self.permutedIndices = [len(self.currentDataFileInfo) - x for x in range (1, len(self.currentDataFileInfo) - (self.size + 1))]
        else:
            raise ValueError("Invalid type.")
    
    def getNewDataPoint(self):
        changedFiles = False
        if self.type == "r":
            while (self.indexAtWithinPermutedIndices >= len(self.permutedIndices)) or (self.indexAtWithinPermutedIndices >= len(self.permutedIndices) * (self.fractionOfTotalDataToUse + .01)):
                self.getNewDataFile()
        elif self.type == "s":
            while self.indexAtWithinPermutedIndices >= len(self.permutedIndices):
                self.getNewDataFile()
                changedFiles = True
        else:
            raise ValueError("Invalid type.")
        index = self.permutedIndices[self.indexAtWithinPermutedIndices]
        toReturn = self.currentDataFileInfo[index : index - self.size : -1], self.currentDataFileInfo[index - self.size]
        self.indexAtWithinPermutedIndices += 1
        #print(index)
        #print(toReturn)
        if self.type == "r":
            return toReturn
        elif self.type == "s":
            return toReturn, changedFiles
        else:
            raise ValueError("Invalid type.")
