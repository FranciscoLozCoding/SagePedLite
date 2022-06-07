from cv2 import sqrt
import jetson.utils
from TrackPerson import Person
import numpy as np
import time
import math

#Camera Variables
camIP = '10.42.0.104'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'

#set up camera and display
camera = jetson.utils.videoSource(streamURL) # Armcrest camera
# camera = jetson.utils.videoSource("file:///home/flozano/Workspace/SagePedLite/example.mp4") # example video

#rtp stream variables
deviceIP = "130.202.141.55"
port = "1234"
rtpStream = "rtp://" + deviceIP + ":" + port

# Video Ouput:
#   * No arguments will display an OpenGL display Window
#   * "file://my_video.mp4" for video file output
#   * "rtp://<remote-ip>:1234" for streaming to another device
#   * by default, an OpenGL display window will be created unless --headless is specified
display = jetson.utils.videoOutput(rtpStream,argv=["--headless"])

# The model does better at detecting objects when it has a leveled view of the objects
# For example if the camera is looking down at the objects the model doesn't do well
import jetson.inference  #Has to be after videoOutput for the X11 window to create
net = jetson.inference.detectNet("pednet",threshold=0.5) #load model

#Make sure the GPU is done work
jetson.utils.cudaDeviceSynchronize()

#initialize font for text
font = jetson.utils.cudaFont( size=20 )

# initialize centroid tracker
from centroidtracker import CentroidTracker
ct = CentroidTracker()
trackableObjects = {}

time.sleep(2.0)

while True:

    img = camera.Capture() #capture image

    detections = net.Detect(img) #run image through model

    centerCords = []

    for x in detections:
        if(net.GetClassDesc(x.ClassID) == "person"):
            centerCords.append(x.Center)
            print("Top: " + str(x.Top))
            print("Bottom: " + str(x.Bottom))
            print("Left: " + str(x.Left))
            print("Right: " + str(x.Right))

    objects = ct.update(centerCords) #update centroid tracker with objects center cordinates

    # loop over the tracked objects
    for (objectID,centroid) in objects.items():

        # check to see if a person exists for the currect object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing person, create one
        if to is None:
            to = Person(objectID, centroid)

            #store the trackable object in our dictionary
            trackableObjects[objectID] = to

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
            to.addCord(centroid)

        #draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        font.OverlayText(img,img.width,img.height,text,centroid[0] - 10,centroid[1] + 10,font.White)
        jetson.utils.cudaDrawCircle(img,centroid,4,(0,255,0))
    
    #loop through the tracked people to draw their tracjectory lines
    for key in trackableObjects:
        i = 0
        while i <= len(trackableObjects[key].centroids) - 4:

            # Get centroids
            centroid1 = trackableObjects[key].centroids[i]
            centroid2 = trackableObjects[key].centroids[i + 3]

            # Get cordinate points
            x1 = centroid1[0]
            x2 = centroid2[0]
            y1 = centroid1[1]
            y2 = centroid2[1]

            # calculate distance of points to see if it is greater than 2
            #  Note: draw line cannot draw a line shorter than 2
            xL = (x1 - x2)**2
            yL = (y1 - y2)**2
            distance = math.sqrt(xL + yL)

            # base line color on direction
            #   * Red is towards camera
            #   * Blue is away from camera
            if y1 > y2 and x1 < x2:
                lineColor = (0,0,255) #blue
            elif y1 < y2 and x1 > x2:
                lineColor = (255,0,0) #red
            elif y1 > y2 and x1 > x2:
                lineColor = (0,0,255) #blue
            elif y1 < y2 and x1 < x2:
                lineColor = (255,0,0) #red

            #Draw line if the distance is greater than 2
            if distance > 2: #Draw Line
                jetson.utils.cudaDrawLine(img,centroid1,centroid2,lineColor,1)
            i += 1

    display.Render(img) #show image
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())) #update window title

    if not camera.IsStreaming() or not display.IsStreaming():
        break