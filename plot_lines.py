import pathlib
import cv2
import numpy as np
from shutil import copyfile
from datetime import datetime
import time
import sqlite3
import os

is_away_from_cam_YLine = 420

#code isn't the most efficient at the moment
#Some improvements: So the hourly and master plots at the same time
#                   Only pull information from the database once
def draw_lines(date):
    #date format: yyyy-mm-dd
    #strip date
    var_date_object = datetime.strptime(date, "%Y-%m-%d") # for individual hour printing
    print(var_date_object)
    master_date_object = str(var_date_object)
    master_date_object = master_date_object.replace(' 00:00:00','') # for keeping track of each day passing through for all line image

    #generic image for overlaying
    #eventually take in name of file or environment number from database for different environments
    im_path = pathlib.Path(os.path.join(os.getcwd(),"Images/env") + "/" + "Env1.jpg")
    image = cv2.imread(str(im_path))
    image_copy = image.copy()
    master_copy = image.copy()

    # Uncomment this section to figure out the line to determine if the person
    # is walking away or towards camera
    ###############################################################################################
    # test = []
    # test.append((1,420))
    # test.append((703,420))
    # master_copy = cv2.polylines(master_copy,np.int32([test]), False, (0,255,239))
    ################################3###############################################################

    #draw lines using the database
    db_path = os.path.join(os.getcwd(),"pedestrian_detections.db")
    db_connection = sqlite3.connect(db_path)
    db_cursor = db_connection.cursor()
    master_date = master_date_object + '%'
    db_cursor.execute("SELECT PERMAID, XCOORD, YCOORD FROM Coordinate WHERE DATE LIKE ?;", (master_date,))
    record = db_cursor.fetchall() # [0] = perma id, [1] = xcoord, [2] = ycoord

    if len(record) < 1: # if record is empty
        print("No records found")
        return # just exit
    
    total_coords = []
    perma_id = record[0][0] # get first perma id for the day

    #hourly images
    for i in range(1,24):
        date_hour = master_date_object + ' ' + str(i) + '%'
        print("Date: ", date_hour)
        db_cursor.execute("SELECT PERMAID, XCOORD, YCOORD FROM Coordinate WHERE DATE LIKE ?;", (date_hour,))
        rec = db_cursor.fetchall() # [0] = perma id, [1] = xcoord, [2] = ycoord
        image_copy = image.copy() #create a copy for the specific hour
        print("Length: ", len(rec))
        if len(rec) < 1:
            continue

        print("rec ", rec)
        print("rec00 ",rec[0][0])
        id = rec[0][0]
        for row in rec:
            if(id != row[0]): #if the ids are different, update colors and write to image
                #update color
                if(total_coords[0][1] > is_away_from_cam_YLine): master_color = (255,0,0)
                else: master_color = (0,0,255)
                image_copy = cv2.polylines(image_copy, np.int32([total_coords]), False, master_color)
                total_coords.clear() #clear coords for single person
                coordinate = (row[1],row[2])
                total_coords.append(coordinate)
                id = row[0] #reset the current id
            else:
                coordinate = (row[1],row[2]) # tuple of the rows coordinates
                total_coords.append(coordinate)

        #update color and write to image for last record where if(id != row[0]) fails
        if(total_coords[0][1] > is_away_from_cam_YLine): master_color = (255,0,0)
        else: master_color = (0,0,255)
        image_copy = cv2.polylines(image_copy, np.int32([total_coords]), False, master_color)

        #write hourly images
        total_coords.clear() # end of record read in
        cv2.putText(image_copy, "Towards Camera", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        cv2.putText(image_copy, "Away from Camera", (10,55), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)
        cv2.putText(image_copy, "Hour: " + str(i), (10,85), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imwrite(os.path.join(os.getcwd(),"Image_label_xmls") + "/" + "crosswalk_detections/" + date + "/line_result_" + str(i) + ".jpg",image_copy)

    #track permaid, store tuples in list, loop through list, for master image containing all points
    for row in record:
        if(row[0] != perma_id):
            perma_id = row[0]
            if(total_coords[0][1] > is_away_from_cam_YLine):
                master_color = (255,0,0)
            else:
                master_color = (0,0,255)
            master_copy = cv2.polylines(master_copy,np.int32([total_coords]), False, master_color)
            total_coords.clear()
            coordinate = (row[1],row[2])
            total_coords.append(coordinate)
        else:
            coordinate = (row[1],row[2]) # tuple of the rows coordinates
            total_coords.append(coordinate)

    #update color and write to image for last record where if(row[0] != perma_id) fails
    if(total_coords[0][1] > is_away_from_cam_YLine):
        master_color = (255,0,0)
    else:
        master_color = (0,0,255)
    master_copy = cv2.polylines(master_copy,np.int32([total_coords]), False, master_color)
    
    db_cursor.close()

    total_coords.clear() # end of record read in

    #write the image for the whole day
    cv2.putText(master_copy, "Towards Camera", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.putText(master_copy, "Away from Camera", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
    cv2.imwrite(os.path.join(os.getcwd(),"Image_label_xmls") + "/" + "crosswalk_detections/" + date + '/line_result_M' + '.jpg', master_copy)

    db_connection.close()   #close db
    return

# for using the script without pedestrian_detection.py, for testing
if __name__ == '__main__':
    draw_lines("2022-06-24")