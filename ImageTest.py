import cv2
import os
import numpy as np

class Game(object):
    def __init__(self):
        self.vegetablePics = []
        self.tomatoPics = []
        self.initVegPics()
        self.initTomPics()
        self.allPics = self.vegetablePics + self.tomatoPics
        self.input = []
        self.print = True

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

    def convertToInputPic(self, source):
        # Convert to HSV
        source = cv2.cvtColor(source,cv2.COLOR_BGR2HSV)
        # Convert to isolate red
        if self.print:
            print(source[0][0])
            self.print = False
        inputPic = []
        for row in range(len(source)):
            newRow = []
            for col in range(len(source[0])):
                hue = source[row][col][0]
                if hue < 18 or hue > 340: newRow.append(1)
                else: newRow.append(0)
            inputPic.append(newRow)
        return inputPic

    def run(self):
        self.y = np.array([[0,0,0,0,0,1,1,1,1,1]]).T

        for i in range(len(self.allPics)):
            print(len(self.allPics[i]), len(self.allPics[i][0]))
            self.input.append(self.convertToInputPic(self.allPics[i]))
        print(self.input[0][0])
game = Game()
game.run()
