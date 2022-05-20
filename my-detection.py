import jetson.utils

#Camera Variables
camIP = '10.42.0.104'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'

#set up camera and display
camera = jetson.utils.videoSource(streamURL)
display = jetson.utils.videoOutput()

import jetson.inference #Has to be after videoOutput for the X11 window to create
net = jetson.inference.detectNet("ssd-mobilenet-v2",threshold=0.5) #load model

while display.IsStreaming():
    img = camera.Capture() #capture image
    detections = net.Detect(img) #run image through model
    display.Render(img) #show image
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())) #update window title