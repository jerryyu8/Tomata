import cv2
import os
import numpy as np

class preProcess(object):
    def __init__(self):
        self.vegetablePics = []
        self.tomatoPics = []
        self.rawTestPics = []
        self.initVegPics()
        self.initTomPics()
        self.initTestPics()
        self.allPics = self.vegetablePics + self.tomatoPics
        self.input = []
        self.output = []
        self.test = []

    def initVegPics(self):
        mainPath = os.getcwd()
        os.chdir("Vegetables")
        for fileName in os.listdir(os.getcwd()):
            self.vegetablePics.append(cv2.imread(fileName))
        os.chdir(mainPath)

    def initTomPics(self):
        mainPath = os.getcwd()
        os.chdir("Tomatos")
        for fileName in os.listdir(os.getcwd()):
            self.tomatoPics.append(cv2.imread(fileName))
        os.chdir(mainPath)

    def initTestPics(self):
        mainPath = os.getcwd()
        os.chdir("TestData")
        count = 0
        for fileName in os.listdir(os.getcwd()):
            print(fileName)
            self.rawTestPics.append(cv2.imread(fileName))
            if count in [1,2]:
                self.rawTestPics[count] = np.transpose(self.rawTestPics[count],(1,0,2))
            count += 1
        os.chdir(mainPath)

    def convertToInputPic(self, source):
        # Convert to HSV
        source = cv2.cvtColor(source,cv2.COLOR_BGR2HSV)
        # Convert to isolate red
        rawInputPic = []
        for row in range(len(source)):
            newRow = []
            for col in range(len(source[0])):
                hue = source[row][col][0]
                if hue < 18 or hue > 340: newRow.append(1)
                else: newRow.append(0)
            rawInputPic.append(newRow)
        # Convert to size of 400x501 image
        sizedInputPic = np.zeros((400,501))
        rawRows, rawCols = len(rawInputPic), len(rawInputPic[0])
        sizedInputPic[:rawRows-400,:rawCols-501] = rawInputPic
        #Reduce it
        inputPic = []
        for oldRow in range(0,400,10):
            createRow = []
            for oldCol in range(0,500,10):
                tempSum = 0
                for newRow in range(10):
                    for newCol in range(10):
                        try: tempSum += sizedInputPic[oldRow+newRow][oldCol+newCol]
                        except: print(oldRow+newRow, oldCol+newCol)
                createRow.append(tempSum/100)
            inputPic.append(createRow)
        # Convert 2D matrix to 1D
        inputPic = np.array(inputPic)
        print(len(inputPic), len(inputPic[0]))
        return inputPic.flatten()


    def run(self):
        # Init Results
        self.output = np.array([[0,0,0,0,0,1,1,1,1,1]]).T
        # Convert all pics
        for i in range(len(self.allPics)):
            # print(len(self.allPics[i]), len(self.allPics[i][0]))
            self.input.append(self.convertToInputPic(self.allPics[i]))
        print(self.input[0][0])
        for i in range(len(self.rawTestPics)):
            # print(len(self.allPics[i]), len(self.allPics[i][0]))
            self.test.append(self.convertToInputPic(self.rawTestPics[i]))

# preProcess = preProcess()
# preProcess.run()
