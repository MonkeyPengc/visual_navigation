# script-5

# author: Cheng Peng
# brief : match keypoints of a flight frame (one json file) with the keypoints of database(json files) via FlannBased matcher
# output: a "summary.json" file, which stores the/ 
# matched info as ((index, (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, desc[index])).

# -----------------------------------------------------------------------------
# import modules

import cv2
import numpy as np
import json
import argparse
import os
import time
import glob
import itertools

# -----------------------------------------------------------------------------
# implementation methods


def KNNmatch(data1, data2, confidence_list, return_sum):
    # Perform KNNmatch algorithm on data
    
    match = np.zeros((len(data1), len(confidence_list)))
    pair_scores = np.zeros((len(data1), 2))
    pair_index = np.zeros((len(data1),2))
    for i in np.arange(len(data1)):
        score = np.zeros((len(data2)))
        for j in np.arange(len(data2)):
            score[j] = sum(np.square(data1[i,:] - data2[j,:]))
        imin0 = np.argmin(score)
        min0 = score[imin0]
        score[imin0] = np.nan
        imin1 = np.nanargmin(score)
        min1 = np.nanmin(score)
        pair_scores[i,:] = min0,min1
        pair_index[i,:] = imin0,imin1
        for n in np.arange(len(confidence_list)):
            if min0 < float(confidence_list[n]) / 100 * min1:
                match[i,n] = 1

    return match, pair_scores, pair_index


class NumpyAwareJSONEncoder(json.JSONEncoder):
   def default(self, obj):
       if isinstance(obj, np.ndarray) and obj.ndim == 1:
           return obj.tolist()
       elif isinstance(obj, np.integer):
           return int(obj)
       elif isinstance(obj, np.floating):
           return float(obj)
       return json.JSONEncoder.default(self, obj)


def write_json_keypoints(keypoint_file, kp, desc):
   data = dict ((index, (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, desc[index])) for index, point in enumerate(kp))

   with open(keypoint_file, "w") as fo:
      fo.write(json.dumps(data, cls=NumpyAwareJSONEncoder) + "\n")


def read_json_keypoints(keypoints_file):
   with open(keypoints_file, "r") as fi:
      feature_dictionary = json.load(fi)
   return feature_dictionary 


def convert_feature_dictionary_to_data_array(feature_dictionary, descriptor_size=64):
    data  = np.zeros((len(feature_dictionary), descriptor_size))
    #print "print feature dictionary", feature_dictionary
    for key in feature_dictionary:
       point_pt, point_size, point_angle, point_response, point_octave, point_class_id, data_list = feature_dictionary[key]
       data[int(key),:] = data_list

    return data


def matcher_flann(desc1, desc2):
   # FLANN matching
   # return total number of matches, number of matches, the corresponding index and scores of matches under
   # different threshold ratios
   
   FLANN_INDEX_KDTREE = 0
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks=50)   # or pass empty dictionary
   matcher = cv2.FlannBasedMatcher(index_params,search_params)
   matches = matcher.knnMatch(np.asarray(desc1,np.float32),np.asarray(desc2,np.float32), k=2)

   good = list()
   vector_count = list()
   kp_index = list()
   kp_score = list()
   
   for threshold in np.arange(0.5,0.8,.05):  # threshold ratios
       threshold_list = list()
       score_list = list()
       for i,(m,n) in enumerate(matches):
           if m.distance < threshold*n.distance:
               good.append(m)
               threshold_list.append(i)
               score_list.append(m.distance)
       vector_count.append(len(threshold_list))
       kp_index.append(threshold_list)
       kp_score.append(score_list)

   return len(matches), vector_count, kp_index, kp_score


def descriptor_matcher(method, desc1, desc2):
   start = time.time()
   if method == "flann":
      # evaluate thresholds to determine confidence 
      length, vector, kp_index, kp_score = matcher_flann(desc1,desc2)

   elif method == "confidence":
      # evaluate thresholds to determine confidence 
      start = time.time()
      confidence_list = np.arange(1,6) * 10
      KNNmatch(desc1, desc2, confidence_list,0)

   end = time.time()
   return length, vector, end-start, kp_index, kp_score


def process_compare(keypoint_file1, keypoint_file2):
   # Get the keypoints/descriptor files
   dictionary1 = read_json_keypoints(keypoint_file1)
   dictionary2 = read_json_keypoints(keypoint_file2)
   
   # key is (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[index])
   #print dictionary1["0"]
   point_pt, point_size, point_angle, point_response, point_octave, point_class_id, data_list = dictionary1["0"]
   descriptor_length = len(data_list)

   # need to convert (descriptor) from dictionary1 and dictionary2 to numpy data arrays.
   desc1 = convert_feature_dictionary_to_data_array(dictionary1, descriptor_length)
   desc2 = convert_feature_dictionary_to_data_array(dictionary2, descriptor_length)
   length, vector, flann_time, kp_index, kp_score = descriptor_matcher("flann", desc1, desc2)

   return length, vector, flann_time, kp_index, kp_score


def compare_flight(flight_file, keypoint_dir, dst_directory, flag_grid, summary_file="summary.json"):
   ## write the matching results into a summary json file
   
   if len(flag_grid) == 1 and flag_grid[0] == 0:  # match with the entire database files
      keypoint_file_list = glob.glob(os.path.join(keypoint_dir,'*.json'))
      
   else:  # or glob a sub-area of keypoints files
      print("***********************")
      keypoint_file_list = list()
      neighbour_grids = list()
      grid_x, grid_y = flag_grid[0], flag_grid[1]
      # get 24 neighbours of the best match
      neighbour_grids = [(grid_x-2, grid_y-2), (grid_x-1, grid_y-2), (grid_x, grid_y-2), (grid_x+1, grid_y-2), (grid_x+2, grid_y-2), (grid_x-2, grid_y-1), (grid_x-1, grid_y-1), (grid_x, grid_y-1), (grid_x+1, grid_y-1), (grid_x+2, grid_y-1), (grid_x-2, grid_y), (grid_x-1, grid_y), (grid_x, grid_y), (grid_x+1, grid_y), (grid_x+2, grid_y), (grid_x-2, grid_y+1), (grid_x-1, grid_y+1), (grid_x, grid_y+1), (grid_x+1, grid_y+1), (grid_x+2, grid_y+1), (grid_x-2, grid_y+2), (grid_x-1, grid_y+2), (grid_x, grid_y+2), (grid_x+1, grid_y+2), (grid_x+2, grid_y+2)]
      
      for x, y in neighbour_grids:
         try: # in case grids out of boundary
            keypoint_file_list.append(glob.glob(os.path.join(keypoint_dir, 'grid_z19_' + str(x) + '_' + str(y) + '.*json'))[0])
         except:
            pass

   compare_list = list(itertools.product([flight_file], keypoint_file_list))
   comparisons = dict((index,(a,b,process_compare(a,b))) for index, (a,b) in enumerate(compare_list))

   with open(os.path.join(dst_directory, summary_file), "w") as fo:
      fo.write(json.dumps(comparisons, cls=NumpyAwareJSONEncoder) + "\n")


def parse_args():
    
   # ----- parse arguments -----
   
   parser = argparse.ArgumentParser(description='matcher')
   parser.add_argument('flight_file', help="Need a flight keypoint file")
   parser.add_argument('directory', help="Need a directory")
   parser.add_argument('dst_directory', help="Need a output directory")
   parser.add_argument('flag_best_grid', nargs='+', type=int)  # if the best match is generated

   args = parser.parse_args()
   flight_file = str(args.flight_file)
   map_keypoint_dir = str(args.directory)
   dst_directory = args.dst_directory
   flag_best_grid = args.flag_best_grid

   # ----- make a directory -----

   if not os.path.exists(dst_directory):
      os.makedirs(dst_directory)

   # implement all
   compare_flight(flight_file, map_keypoint_dir, dst_directory, flag_best_grid)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
   parse_args()


