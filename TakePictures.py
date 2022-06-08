import os
import cv2 #video streaming
from datetime import datetime
import time

#Time variable
seconds_between_frames = 1.0 #seconds between the frames captured to jpg

#Camera Variables
camIP = '10.42.0.104'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'
cam = cv2.VideoCapture(streamURL)

#path to image folder
Image_Base_Path = os.path.join(os.getcwd(),"Images")

#Capture images and show stream until esc key is pressed
while True:
    #show stream
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Image", frame)

    #capture images with name based on datetime and organized by date
    img_dateTime = datetime.now()
    month = "%02d" % (img_dateTime.month)
    day = "%02d" % (img_dateTime.day)
    year = "%04d" % (img_dateTime.year)

    #create directory if it doesnt exist
    if not os.path.isdir(Image_Base_Path + "/" + year): #create yyyy/mm/dd folder if yyyy folder doesnt exist
        os.makedirs(Image_Base_Path + "/" + year + "/" + month + "/" + day)
    elif not os.path.isdir(Image_Base_Path + "/" + year + "/" + month): #create yyyy/mm/dd folder if yyyy/mm folder doesnt exist
        os.makedirs(Image_Base_Path + "/" + year + "/" + month + "/" + day)
    elif not os.path.isdir(Image_Base_Path + "/" + year + "/" + month + "/" + day): #create yyyy/mm/dd folder if yyyy/mm/dd folder doesnt exist
        os.makedirs(Image_Base_Path + "/" + year + "/" + month + "/" + day)

    #path to the image taken
    Image_Path = os.path.join(Image_Base_Path,year,month,day)
    
    #Create image
    img_name="{}.jpg".format(img_dateTime)
    img_path=os.path.join(Image_Path,img_name)
    cv2.imwrite(img_path, frame)

    #get key pressed
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    time.sleep(seconds_between_frames)

#closes camera
cam.release()

#close window
cv2.destroyAllWindows()