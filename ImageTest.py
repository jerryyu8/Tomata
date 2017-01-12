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

    def run(self):
        self.y = np.array([[0,0,0,0,0,1,1,1,1,1]]).T
        print(self.allPics[0])
        print('run')

game = Game()
game.run()
