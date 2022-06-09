import jetson.utils
from datetime import datetime
import os
import time
import jetson.inference
from pascal_voc_writer import Writer


# The model does better at detecting objects when it has a leveled view of the objects
# For example if the camera is looking down at the objects the model doesn't do well
print('Loading model...',end='')
start_time = time.time()
net = jetson.inference.detectNet("pednet",threshold=0.5) #load model
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


#Get the images captured by the EDU node
IMAGE_PATHS = []

# Check to see if the user wants a specific data ran, otherwise the default is the current date
import sys
output_dir_date = ""
if len(sys.argv) > 2: #wrong number of arguments
    print("\n Format: python plot_object_detection.py date")
    print("Where date = yyyy/mm/dd ")
    quit()
if len(sys.argv) == 2: # has an input date
    current_date = sys.argv[1] + "/"
    output_dir_date = sys.argv[1]
    new_dir = sys.argv[1]
    new_dir = new_dir.split("/")
    output_dir_date = new_dir[1] + "-" + new_dir[2] + "-" + new_dir[0] + "/" #reformatting to output format "mm-dd-yyyy"
now = datetime.now()
month = "%02d" % (now.month)
day = "%02d" % (now.day)
year = "%04d" % (now.year)
if len(sys.argv) == 1: # default to current date
    current_date = year + "/" + month + "/" + day + "/"
    output_dir_date =  month + "-" + day + "-" + year + "/"  #reformatting to output format "mm-dd-yyyy"

temp_image_dir =  os.path.join(os.getcwd(),"Images") + "/" + current_date
xml_output_dir = os.path.join(os.getcwd(),"Image_label_xmls") + "/" + output_dir_date # directory is created if it does not exist later

for filename in os.listdir(temp_image_dir): # raw image directory
    if os.path.getsize((os.path.join(temp_image_dir + filename))) <= 0:
        print("Corrupted file found, terminating program")
        quit()
    IMAGE_PATHS.append(os.path.join(temp_image_dir + filename))

############################################################################################
# detect_fn() function: Detect pedestrians in image
# @input image the picture to analyze
# @output detections a list of detected pedestrians
###########################################################################################
def detect_fn(image):

    detections = net.Detect(image) #run image through model

    return detections

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') #suppress Matplotlib warnings

############################################################################################
# load_image_into_numpy_array() function: load image from file into numpy array
# @input path the image path
# @output numpy array
###########################################################################################
def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


file_hour = 13
last_hour = file_hour
file_date = " "


############################################################################################
# write_label_xmls() function: create xml annotation file for image that contains image details such as objects detected
# @input image_path the image path
# @output var_time_object.hour the hour the image was taken
# @output file_date the date the image was taken
# @output bool true if the file was processed otherwise false
###########################################################################################
def write_label_xmls(image_path):

    file_name = os.path.basename(image_path)
    var_date_time = file_name[:len(file_name)-4].split(" ")
    var_date_str, var_time_str = var_date_time[0], var_date_time[1]
    var_time_object = datetime.strptime(var_time_str,"%H:%M:%S")
    var_date_object = datetime.strptime(var_date_str,"%Y-%m-%d")

    file_date = str(var_date_object.year) + "/" + "{:02d}".format(var_date_object.month) + "/" + "{:02d}".format(var_date_object.day)

    xml_output_dir = os.path.join(os.getcwd(),"Image_label_xmls") + "/" + "{:02d}".format(var_date_object.month) + "-" + "{:02d}".format(var_date_object.day) + "-" + str(var_date_object.year) + "/"


    # specify the hours of the day you wish to run
    if var_date_object.day > 0 and var_date_object.month >= 0 and 13 <= var_time_object.hour <= 22: # change this 13 to run more than 1 hour
        image_np = np.array(Image.open(image_path))
        # actual detection
        output_dict = detect_fn(Image.open(image_path))
        im_width, im_height = image_np.shape[1], image_np.shape[0]

        for data in output_dict:
            ClassId = data.ClassID
            left = data.Left * im_width
            right = data.Right * im_width
            top = data.Top * im_height
            bottom = data.Bottom * im_height
            score = data.Confidence
            writer = Writer(image_path, im_width, im_height)

            if score >= .5: # .5 Threshold for accuracy on xml objects
                name = net.GetClassDesc(ClassId)
                writer.addObject(name,int(left),int(top),int(right),int(bottom))

        path_to_xml = xml_output_dir + os.path.basename(str(image_path)).replace("jpg","xml") #get the xml files
        writer.save(path_to_xml)
        print("Done: ", os.path.basename(path_to_xml))
        print("Hour: ", file_hour)

        return var_time_object.hour, file_date, True #return true if the file was processed

    return var_time_object.hour, file_date, False #return false if the file was not processed

#creation of xml output directory if it does not exist
if not os.path.isdir(xml_output_dir): 
    os.mkdir(xml_output_dir)

from subprocess import Popen
import pedestrian_detection

count = 0

# For running pedestrian_detection.py
for image_path in IMAGE_PATHS: # add the .xml files into the correct directories
    xmp_path = xml_output_dir + os.path.basename(str(image_path)).replace("jpg","xml")
    file_hour, file_date, processed = write_label_xmls(image_path)
    if file_hour != last_hour and file_hour >= 13 and processed: #hour has changed
        pedestrian_detection.main(last_hour,file_date, False, count==0)
        last_hour = file_hour
        count += 1

#Runs pedestrian_detection.py with "plot" set to true so it runs plot_lines.py
pedestrian_detection.main(last_hour,file_date, True, False)


            

