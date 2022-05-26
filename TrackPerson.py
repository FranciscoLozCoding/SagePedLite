#import the necessary packages
import numpy as np

class Person:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        # initialize dictionaries to store the timestamp and
        # position of the object at various points
        self.timestamp = {"A": 0,"B": 0,"C": 0,"D": 0}
        self.position = {"A": None,"B": None,"C": None,"D": None}
        self.lastPoint = False

        #initialize boolean to indicate if the trajectory has been logged
        self.logged = False

        #Initialize the direction of the object
        self.direction = None
    
    def addCord(self, centroid):
        self.centroids.append(centroid)