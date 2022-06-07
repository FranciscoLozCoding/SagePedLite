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
# @input frame_queue ????
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
# @input frame_queue a collections of frames in queue
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