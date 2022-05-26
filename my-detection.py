import jetson.utils
from TrackPerson import Person
import numpy as np

#Camera Variables
camIP = '10.42.0.78'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'

#set up camera and display
camera = jetson.utils.videoSource(streamURL)
display = jetson.utils.videoOutput("file://my_video.mp4")

#Has to be after videoOutput for the X11 window to create
import jetson.inference 
net = jetson.inference.detectNet("ssd-mobilenet-v2",threshold=0.5) #load model

#Make sure the GPU is done work
jetson.utils.cudaDeviceSynchronize()

#initialize font for text
font = jetson.utils.cudaFont( size=32 )

# initialize centroid tracker
from centroidtracker import CentroidTracker
ct = CentroidTracker()
trackableObjects = {}
people = []

while True:

    img = camera.Capture() #capture image

    detections = net.Detect(img) #run image through model

    centerCords = []

    for x in detections:
        if(net.GetClassDesc(x.ClassID) == "person"):
            centerCords.append(x.Center)

    objects = ct.update(centerCords) #update centroid tracker with objects center cordinates

    # loop over the tracked objects
    for (objectID,centroid) in objects.items():

        # check to see if a person exists for the currect object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing person, create one
        if to is None:
            to = Person(objectID, centroid)
            people.append(to)
        #log the direction and tracjectory of the person walking
        elif to.lastPoint:
            # check if the direction of the object has been set, if
            # not, calculate it, and set it
            if to.direction is None:
                y = [c[0] for c in to.centroids]
                direction = centroid[0] - np.mean(y)
                to.direction = direction
        # otherwise collect new cord
        else:
            for x in people:
                if x.objectID == objectID:
                    x.addCord(centroid)

        #draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        font.OverlayText(img,img.width,img.height,text,centroid[0] - 10,centroid[1] + 10,font.White)
        jetson.utils.cudaDrawCircle(img,centroid,4,(0,255,0))

    display.Render(img) #show image
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())) #update window title

    if not camera.IsStreaming() or not display.IsStreaming():
        break