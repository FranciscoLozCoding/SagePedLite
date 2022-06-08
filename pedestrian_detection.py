import enum
import os
import pathlib
import cv2
import numpy as np
import operator
from datetime import datetime
from shutil import copyfile
import xml.etree.ElementTree as ET
import sys
import math
from numpy.core.fromnumeric import var

sys.path.insert(1,'./deep-person-reid/')
import torch
import torchreid
from torchreid.utils import FeatureExtractor
from collections import defaultdict,deque
from recordtype import recordtype
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import sympy
import pickle
import time

count = 0
second_count = 0
person_id=1
total_person_count=0
frame_id=0
frame_counter=0
frame_queue=deque([],5) # keeps track of previos 5 frames - useful for re-id
person_pos = dict()
dict_person_crossed_the_road = dict()
dict_person_use_the_crosswalk = dict()
dict_person_assigned_number_frames = dict()
dict_frame_time_stamp = dict()
max_person_count=0
   
# GLOBAL VARIABLES TO FIND LINES ON THE ROAD
# Slopes found using (y2-y1)/(x2-x1) - points found by looking at road, in standard form ax^2 + bx + c = 0
north_road_slope = 0.037109375
north_ycoord = 830
south_road_slope = 0.0882352941176
south_ycoord = 1025

# CONTAINS MODEL - IF WANTING TO CHANGE MODELS CHANGE THE "model_name" variable
extractor = FeatureExtractor(model_name='osnet_x1_0', model_path='./model.pth.tar',device='cuda')

############################################################################################
# get_crosswalk_coordinates() function: Gets the crosswalk coordinates 
#   (the coordinates are slightly bigger than the exact coordinates from corner to corner)
# @output numpy array containing cords
# @Note IN THE FUTURE: try implementing a model to detect crosswalks so the cords wont be 
#   hardcoded in. This will enable easy set up when moving the camera to a new location
###########################################################################################
def get_crosswalk_coordinates():
    coordinates = [[514,796],[721,783],[1095,934],[763,992]]
    return np.array(coordinates)

############################################################################################
# get_highlightable_coordinates() function: Gets the exact crosswalk coordinates 
#   (used for highlighting later on in the script, not the actual detections) 
# @output numpy array containing cords
# @Note IN THE FUTURE: try implementing a model to detect crosswalks so the cords wont be 
#   hardcoded in. This will enable easy set up when moving the camera to a new location
###########################################################################################
def get_highlightable_coordinates():
    coordinates = [[524,802],[667,790],[1023,941],[758,962]] # for highlighting the crosswalk
    return np.array(coordinates)

############################################################################################
# parse_xml() function: parse the xml file
# @input xml_file the xml file to be parsed
# @output array containing person objects in the xml file
###########################################################################################
def parse_xml(xml_file):
    final_arr=[]
    tree= ET.parse(xml_file)
    root = tree.getroot()
    for object in root.findall('object'):
        arr=[]
        for box in object.find('bndbox'):
            if object.find('name').text == 'person':
                arr.append(int(box.text))
        if len(arr) > 0:
            final_arr.append(arr)
    return final_arr

############################################################################################
# non_max_suppression_fast() function: multiple boxes per object, compresses into just one
#    box for the object. Finds overlap between pictures, if there are no object boxes, skip
#    the picture
# @input boxes 
# @input overlapThresh threshold of overlap
# @output boxes 
###########################################################################################
def non_max_suppression_fast(boxes,overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick =[]
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bouding
    # boxes by the bottom-right y-cord of the bouding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) cords for the start of the
        # bounding box and the smallest (x, y) cords
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the integer data type
        return boxes[pick].astype("int")

############################################################################################
# get_total_person_count() function: Get the total person count
# @output integer the total person count 
###########################################################################################
def get_total_person_count(current_frame_persons):
    global dict_person_assigned_number_frames
    try:
        return max(dict_person_assigned_number_frames)
    except:
        return 0

############################################################################################
# update_current_frame_assignments() function: ????
# @input current_frame_persons ????
# @input current_frame_sim_score ????
# @input max_score ???
# @input max_person_id ????
# @input best_match_number ????
# @input frame_queue Keeps track of previous 6 frames
# @output current_frame_persons ????
# @output current_frame_sim_score ????
###########################################################################################
def update_current_frame_assignments(current_frame_persons, current_frame_sim_score,max_score, max_person_id,best_match_number, frame_queue):

    if max_person_id in current_frame_sim_score:
        del current_frame_sim_score[max_person_id]

    if max_person_id == -1:
        for current_id, current_person in enumerate(current_frame_persons):
            if current_person.assigned_number == 0:
                current_frame_persons[current_id].assigned_number = get_total_person_count(
                    current_frame_persons)+1 #ADD 1 TOTAL PERSON COUNT BUT DOESN'T UPDATE TOTAL
                if current_person.assigned_number in dict_person_assigned_number_frames:
                    dict_person_assigned_number_frames[current_frame_persons[current_id].assigned_number].append(current_person.frame_id)
                else:
                    dict_person_assigned_number_frames[current_frame_persons[current_id].assigned_number] == []
                    dict_person_assigned_number_frames[current_frame_persons[current_id].assigned_number].append(current_person.frame_id)
    
    for current_id, current_person in enumerate(current_frame_persons):
        if current_person.person_id == max_person_id:
            within_range = True
            for frame in frame_queue:
                for person in frame.person_records:
                    if person.assigned_number == best_match_number:
                        within_range = check_proximity(person.center_cords,current_person.center_cords)
            if max_score > 0.6 and within_range:
                current_frame_persons[current_id].assigned_number = best_match_number
            else:
                current_frame_persons[current_id].assigned_number = get_total_person_count(current_frame_persons)+1
                if current_person.assigned_number in dict_person_assigned_number_frames:
                    dict_person_assigned_number_frames[current_person.assigned_number].append(current_person.frame_id)
                else:
                    dict_person_assigned_number_frames[current_person.assigned_number] = []
                    dict_person_assigned_number_frames[current_person.assigned_number].append(current_person.frame_id)

    for person_id, scores in list(current_frame_sim_score.items()):
        for k,v in list(current_frame_sim_score[person_id].items()):
            if k == best_match_number:
                del current_frame_sim_score[person_id][k]

    return current_frame_persons,current_frame_sim_score

############################################################################################
# is_all_current_frame_persons_assigned() function: checks to see if every person has an ID
# @input current_frame_persons the number of current people in the frame
# @output bool true if every person has an ID otherwhise false
###########################################################################################
def is_all_current_frame_persons_assigned(current_frame_persons):
    
    for current_id, current_person in enumerate(current_frame_persons):
        if current_person.assigned_number == 0:
            return False
    return True

############################################################################################
# find_best_match_score() function: find the best match for the person to re identify
# @input current_frame_persons the amount of people in the current frame
# @input current_frame_sim_score ????
# @input total_person_count ????
# @input frame_queue Keeps track of previous 6 frames
# @output current_frame_persons ????
# @output current_frame_sim_score ????
###########################################################################################
def find_best_match_score(frame_queue, current_frame_persons, current_frame_sim_score, total_person_count):

    while(not is_all_current_frame_persons_assigned(current_frame_persons)):
        max_score, max_score_person_id, best_match_number = 0, -1, -1
        for person_id, scores in list(current_frame_sim_score.items()):

            sim_score = current_frame_sim_score[person_id]

            if len(sim_score) > 0 and max (sim_score.values()) > max_score:
                max_score = max(sim_score.values()) #get max value in sim_score -> sim_score is all similarity scores b/w person Id and other people
                max_score_person_id = person_id
                best_match_number = max(sim_score, key=sim_score.get)
        
        current_frame_persons, current_frame_sim_score = update_current_frame_assignments(
            current_frame_persons, current_frame_sim_score, max_score, max_score_person_id, best_match_number, frame_queue)
    
    frame_queue, current_frame_persons = update_previous_frame(frame_queue, current_frame_persons)

    return current_frame_persons, frame_queue

############################################################################################
# update_previous_frame() function: ????
# @input frame_queue Keeps track of previous 6 frames
# @input current_frame_persons ????
# @output frame_queue Keeps track of previous 6 frames
# @output current_frame_persons ????
###########################################################################################
def update_previous_frame(frame_queue, current_frame_persons):

    arr=[]
    for frame_id, previous_frame in enumerate(frame_queue):
        for person_id, previous_person in enumerate(previous_frame.person_records):
            found = False
            if previous_person.assigned_number in dict_person_assigned_number_frames:
                if len(dict_person_assigned_number_frames[previous_person.assigned_number]) == 1:
                    for current_person in current_frame_persons:
                        if current_person.assigned_number == previous_person.assigned_number:
                            found = True
                    if found is False:
                        arr.append([previous_person.assigned_number, frame_id, person_id])
    if len(arr)>0:
        for val in arr:
            del frame_queue[val[1]].person_records[val[2]]
        for person_id, current_person in enumerate(current_frame_persons):
            if current_person.assigned_number > val[0]:
                current_frame_persons[person_id].assigned_number -= len(arr)
    return frame_queue, current_frame_persons

############################################################################################
# assign_numbers_to_person() function: Assigns a number or a sim score to a person
# @input frame_queue Keeps track of previous 6 frames
# @input current_frame_persons people in the current frame
# @input total_person_count the total count of people
# @output returns things but never actually does anything with them
###########################################################################################
def assign_numbers_to_person(frame_queue, current_frame_persons, total_person_count):

    cos = torch.cosine_similarity(dim=1,eps=1e-6)

    if not any(frame_queue):

        for current_id, current_person in enumerate(current_frame_persons):

            total_person_count+=1
            current_frame_persons[current_id].assigned_number = total_person_count
        return current_frame_persons
    else:

        current_frame_sim_score=dict()
        for current_id, current_person in enumerate(current_frame_persons):
            sim_score = defaultdict(list)
            for previous_frame in frame_queue:
                for previous_person in previous_frame.person_records:
                    similarity_score = cos(current_person.feature,previous_person.feature).data.cpu().numpy() # or .cpu().numpy()
                    sim_score[previous_person.assigned_number].append(similarity_score) # assigns a sim score to the prev person's assigned number
            for assigned_number in sim_score:
                sim_score[assigned_number] = np.mean(sim_score[assigned_number])
            current_frame_sim_score[current_person.person_id] = sim_score

        return find_best_match_score(frame_queue, current_frame_persons, current_frame_sim_score, total_person_count)

############################################################################################
# update_person_position_and_frame() function: Adds the person position and the frame they're
#   located in to the dictionaries
# @input current_frame_persons people in the current frame
# @input current_frame_persons the person's position
# @input current_frame_id the id of the frame
# @output the person's position
###########################################################################################
def update_person_position_and_frame(current_frame_persons, person_pos, current_frame_id):

    for current_person in current_frame_persons:
        if current_person.assigned_number not in person_pos:
            person_pos[current_person.assigned_number] = []
            person_pos[current_person.assigned_number].append(current_person.center_cords)
        else:
            person_pos[current_person.assigned_number].append(current_person.center_cords)

        if current_person.assigned_number in dict_person_assigned_number_frames:
            dict_person_assigned_number_frames[current_person.assigned_number].append(current_person.frame_id)
        else:
            dict_person_assigned_number_frames[current_person.assigned_number] = []
            dict_person_assigned_number_frames[current_person.assigned_number].append(current_person.frame_id)
   
    return person_pos

############################################################################################
# update_person_frame() function: ????
# @input frame_queue Keeps track of previous 6 frames
# @input person_pos Dictionary to hold coordinates of people
# @input current_frame_id the id of the frame
# @output person_pos Dictionary to hold coordinates of people
# @output frame_queue Keeps track of previous 6 frames
###########################################################################################
def update_person_frame(current_frame_id, frame_queue, person_pos):

    for assigned_number in list(dict_person_assigned_number_frames.keys()):
        arr = dict_person_assigned_number_frames[assigned_number]
        if len(arr) == 1:
            if current_frame_id - arr[0] > 0:
                del dict_person_assigned_number_frames[assigned_number]
                if assigned_number in dict_person_crossed_the_road:
                    del dict_person_crossed_the_road[assigned_number]
                if assigned_number in dict_person_use_the_crosswalk:
                    del dict_person_use_the_crosswalk[assigned_number]
                if assigned_number in person_pos:
                    del person_pos[assigned_number]
                for frame_id, previous_frame in enumerate(frame_queue):
                    for person_id, previous_person in enumerate(previous_frame.person_records):
                        if previous_person.assigned_number == assigned_number:
                            del frame_queue[frame_id].person_records[person_id]

        elif len(arr) == 2:
            if arr[1]-arr[0] > 1:

                del dict_person_assigned_number_frames[assigned_number]
                if assigned_number in dict_person_crossed_the_road:
                    del dict_person_crossed_the_road[assigned_number]
                if assigned_number in dict_person_use_the_crosswalk:
                    del dict_person_use_the_crosswalk[assigned_number]
                if assigned_number in person_pos:
                    del person_pos[assigned_number]
                for frame_id, previous_frame in enumerate(frame_queue):
                    for person_id, previous_person in enumerate(previous_frame.person_records):
                        if previous_person.assigned_number == assigned_number:
                            del frame_queue[frame_id].person_records[person_id]

    return frame_queue, person_pos

############################################################################################
# middle_between_points() function: Returns the midle of two (x,y) cordinates
# @input point1 
# @input point2 
# @output middle (x,y) cord being the middle of two points
###########################################################################################
def middle_between_points(point1, point2):
    middle = [None] * 2
    middle[0] = (point1[0] + point2[0]) // 2 #x cord
    middle[1] = (point1[1] + point2[1]) // 2 #y cord

    return middle

############################################################################################
# check_proximity() function: Checks the proximity between 2 points within a set threshold
# @input point1 
# @input point2 
# @output bool true if within proximity
###########################################################################################
def check_proximity(point1, point2):
    y_thresh = 250
    delta_y = point1[1] - point2[1]
    if delta_y <= y_thresh or delta_y >= -y_thresh:
        x_thresh = 500
        delta_x = point1[0] - point2[0]
        if delta_x <= x_thresh or delta_x >= -x_thresh:
            return True
    return False

def did_person_cross_the_road(assigned_number, person_pos):
    #crossing the road conditions
    north_side = False #condition_1
    south_side = False #condition_2
    #values for each side of the road, change these for new images

    #get crosswalk coords
    crosswalk_coords = get_crosswalk_coordinates()
    center_top = middle_between_points(crosswalk_coords[0], crosswalk_coords[1])
    center_bottom = middle_between_points(crosswalk_coords[2], crosswalk_coords[3])

    arr = []
    if assigned_number in person_pos:
        current_person_pos = person_pos[assigned_number]
        for cords in current_person_pos:
            #print(cords)
            if (north_road_slope*cords[0])+cords[1]-north_ycoord < 0: # use middle of road
                north_side = True
            if (south_road_slope*cords[0])+cords[1]-south_ycoord > 0:
                south_side = True
            if (north_road_slope*cords[0])+cords[1]-north_ycoord > 0 and (south_road_slope*cords[0])+cords[1]-south_ycoord < 0: #use middle of sidewalk
                arr.append(cords)
    if len(arr) > 1:
        distance_covered = float(Point(arr[0]).distance(Point(arr[-1])))
        total_distance = float(Point(center_top).distance(Point(center_bottom)))
        pct = distance_covered/ total_distance
        if (north_side and south_side) or (north_side and pct>0.8) or (south_side and pct>0.8):
            return True
    return False

############################################################################################
# angle_between_crosswalk_and_trajectory() function: Calculates the angle of the crosswalk
#   and the tracjetory of the person
# @input person_pos Dictionary to hold coordinates of people
# @output the angle
###########################################################################################
def angle_between_crosswalk_and_trajectory(person_pos):
    import math
    from sympy import Point, Line, pi

    #get crosswalk coords
    crosswalk_coords = get_crosswalk_coordinates()
    #get center of top of crosswalk
    center_top = middle_between_points(crosswalk_coords[0], crosswalk_coords[1])
    #get center of bottom
    center_bottom = middle_between_points(crosswalk_coords[2], crosswalk_coords[3])

    ne=sympy.Line(center_top,center_bottom)
    arr=[]
    angle=[]
    for cords in person_pos:
        if (north_road_slope*cords[0])+cords[1]-north_ycoord > 0 and (south_road_slope*cords[0])+cords[1]-south_ycoord < 0:
            arr.append(cords)
    for pair_id, val in enumerate(arr):
        if pair_id < len(arr)-1:
            angle.append(math.degrees(sympy.Line((arr[pair_id][0],arr[pair_id][1]), (arr[pair_id+1][0],arr[pair_id+1][1])).
                                  angle_between(ne)))
    # print(angle)
    return angle

############################################################################################
# did_person_use_the_crosswalk() function: checks if person used crosswalk within a certain 
#   polygon
# @input person_cords 
# @input crosswalk_cords 
# @output bool true if the person used the crosswalk
###########################################################################################
def did_person_use_the_crosswalk(person_cords, crosswalk_cords):
    count=0
    # using the crosswalk coordinates
    crosswalk_polygon = Polygon([(524,802),(667,790),(1023,941),(758,962)])
    #print("Pratool person cords",person_cords)
    for cords in person_cords:
        if crosswalk_polygon.contains(Point(cords)):
            count+=1
    if count>3:
        return True
    return False

############################################################################################
# color_the_person_box() function: Color the person box based on trajectory
# @input img_original The original image being processed (image)
# @input assigned_number The person's assigned number (int)
# @input person_pos Dictionary to hold coordinates of people
# @input person_cords Unused (x,y pair)
# @input crosswalk_cords Coordinates of the crosswalk
# @input val bounding box of person (x min,max y min, max)
# @input dict_person_crossed_the_road  Dictionary to check if the person has crossed the 
#   roads or not, person is key
# @input dict_person_use_the_crosswalk Dictionary to check if the person has crossed the 
#   crosswalk or not, person is key
# @output img_original
# @output dict_person_crossed_the_road  Dictionary to check if the person has crossed the 
#   roads or not, person is key
# @output dict_person_use_the_crosswalk Dictionary to check if the person has crossed the 
#   crosswalk or not, person is key
###########################################################################################
def color_the_person_box(img_original, assigned_number, person_pos, person_cords, crosswalk_cords, 
                         val,dict_person_crossed_the_road, dict_person_use_the_crosswalk):
    if did_person_cross_the_road(assigned_number, person_pos):
        #angle = angle_between_crosswalk_and_trajectory(person_pos[assigned_number])
        if assigned_number not in dict_person_crossed_the_road:
            dict_person_crossed_the_road[assigned_number] = True
           
        if did_person_use_the_crosswalk(person_cords, crosswalk_cords):# and any(x<25 or x>155 for x in angle):
            if assigned_number not in dict_person_use_the_crosswalk:
                dict_person_use_the_crosswalk[assigned_number] = True
              
            cv2.rectangle(img_original,(val[1],val[2]),(val[3],val[4]),(0,255,0),5)#green 
        else:
            cv2.rectangle(img_original,(val[1],val[2]),(val[3],val[4]),(0,0,255),5)#red
    else:
        cv2.rectangle(img_original,(val[1],val[2]),(val[3],val[4]),(255,255,255),3)#white
        
    return img_original, dict_person_crossed_the_road, dict_person_use_the_crosswalk


#For standalone use: All functionality of pedestrian detection script should remain intact,
# even when the script is done being modified to work in real time
def main(interval = -1, date = None, plot = False, initial=True):

    image_list=[] # Array the hold the new images created from this script
    date_arr=[]
    new_file_path = ""

    # Made everything global because it needs to be imported in plot_object_detection so we can keep data across hours/whole day
    global dict_person_assigned_number_frames, dict_person_crossed_the_road, dict_person_use_the_crosswalk, dict_frame_time_stamp
    global count
    global second_count
    global person_id
    global total_person_count
    global frame_id
    global frame_counter
    global frame_queue
    global person_pos
    global max_person_count

    if initial:
        count = 0
        second_count = 0
        person_id=1
        total_person_count=0
        frame_id=0
        frame_counter = 0
        frame_queue = deque([],6)              # Keeps track of previous 6 frames - used for re-id
        person_pos = dict()                    # Dictionary to hold coordinates of people
        dict_person_crossed_the_road = dict()  # Dictionary to check if the person has crossed the roads or not, person is key
        dict_person_use_the_crosswalk = dict() # Dictionary to check if the person has crossed the crosswalk or not, person is key
        dict_person_assigned_number_frames = dict() 
        dict_frame_time_stamp = dict()
        max_person_count=0

    
    size = (0,0)    # Used in creating a .mp4 video at the end of the script

    frame_record = recordtype("frame_record", "frame_id person_records")
    person_record = recordtype("person_record", "person_id frame_id feature assigned_number center_cords bottom_cords")
    pts = get_highlightable_coordinates()# Uses exact crosswalk coordinates as a highlighter for visual aid

    hour_min = 13   #default hour range
    hour_max = 22

    # Allows user to run the script through command line arguments (.xml files must exist)
    if len(sys.argv) < 2 and interval == -1:
        print("\n \nFormat: python pedestrian_detection.py [hour_min] [hour_max] [date1, date2, ...]")
        print("Where hour_min / hour_max = the hour range, dateN = yyyy/mm/dd ")
        print("If times are not found, will run hours between 13 and 22.")
        return
