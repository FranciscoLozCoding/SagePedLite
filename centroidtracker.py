#import packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    ############################################################################
    # Constructor:
    # @var self the object itself
    # @var maxDisappeared the maximum number of consecutive frames an object
    #        has to be lost until it is removed from the tracker
    #############################################################################
    def __init__(self,maxDisappeared=20):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        
    ############################################################################
    # register method: 
    #   responsible for adding new objects to our tracker
    # @var self the object itself
    # @var centroid the center coordinate of the bounding rectangle
    #############################################################################
    def register(self,centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

        
    ############################################################################
    # deregister method: 
    #   responsible for removing objects from our tracker
    # @var self the object itself
    # @var ObjectId the id of the object being tracked
    #############################################################################
    def deregister(self,ObjectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[ObjectID]
        del self.disappeared[ObjectID]
        
    ############################################################################
    # update method: 
    #   responsible for updating objects from our tracker with a new centroid location
    #   that minimizes the Euclidean distance from the old centroid
    # @var self the object itself
    # @var centerCords a list of coordinates for all object centers
    # @return objects ordered dictionary of object Ids
    #############################################################################
    def update(self,centerCords):
        # check to see if the list of input center cords is empty
        if len(centerCords) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                #if we have reached a maximum number of consecutive
                #frames where a given object has been markes as missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info to update
            return self.objects
        
        # Initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(centerCords),2),dtype="int")

        #loop over the center cords
        for (i,(CenterX,CenterY)) in enumerate(centerCords):
            #use the center cords to derive the centroid
            cX = int(CenterX)
            cY = int(CenterY)
            inputCentroids[i] = (cX,cY)
        
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0,len(inputCentroids)):
                self.register(inputCentroids[i])
        # otherwise, we need to update any existing objects by trying to match
        # the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids),inputCentroids)

            # in order to perform this matching we must 1) find the 
            # smallest value in each row and then 2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index list
            rows = D.min(axis=1).argsort()

            # next, we perform a similiar process on the columns by 
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register, or
            # deregister an object we need to keep track of which 
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row,column) index tuples
            for (row,col) in zip(rows,cols):
                #if we have already examined either the row or column
                # value before, ignore it 
                if row in usedRows or col in usedCols:
                    continue

                #Otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                #indicate that we have examined each of the row and column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            #compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0,D.shape[0])).difference(usedRows)
            unusedCols = set(range(0,D.shape[1])).difference(usedCols)

            # in the event that number of object centroids is equal or greater than the number of input centroids
            # we need to check and see if some of these objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater than the number of existing object centroids
            # we need to register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        
        # return the set of trackable objects
        return self.objects
