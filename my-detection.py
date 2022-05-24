#import cv2
import jetson.utils

#Camera Variables
camIP = '10.42.0.103'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'

#set up camera and display
camera = jetson.utils.videoSource(streamURL)
display = jetson.utils.videoOutput()

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

#trajectory variables
# import numpy as np
# trajectory = np.empty([2,2])

while display.IsStreaming():

    img = camera.Capture() #capture image

    detections = net.Detect(img) #run image through model

    centerCords = []

    for x in detections:
        if(net.GetClassDesc(x.ClassID) == "person"):
            centerCords.append(x.Center)
    #         trajectory = np.append(trajectory,[x.Center],axis=0)
    #         img = cv2.polylines(img,trajectory,False,(0,0,255))
    objects = ct.update(centerCords) #update centroid tracker with objects center cordinates

    # loop over the tracked objects
    for (objectID,centroid) in objects.items():
        #draw both the ID of the object and the centroid of the object on the output frame
        text = "ID {}".format(objectID)
        font.OverlayText(img,img.width,img.height,text,centroid[0] - 10,centroid[1] + 10,font.White)
        jetson.utils.cudaDrawCircle(img,centroid,4,(0,255,0))

    display.Render(img) #show image
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())) #update window title