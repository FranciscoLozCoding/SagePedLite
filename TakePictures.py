import os
import cv2 #video streaming
import datetime

camIP = '10.42.0.103'
username = 'admin'
password = 'admin'
streamURL = 'rtsp://' + username + ':' + password + '@' + camIP + ':554/cam/realmonitor?channel=1&subtype=1'

cam = cv2.VideoCapture(streamURL)
Path = os.path.join(os.getcwd(),"Images")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Image", frame)

    img_date = datetime.datetime.now()
    img_name="frame_{}.png".format(img_date)
    img_path=os.path.join(Path,img_name)
    cv2.imwrite(img_path, frame)
    print("{} written!".format(img_name))

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()