
# brief : keypoints files self-matching

# -----------------------------------------------------------------------------
# import modules

import cv2
import numpy as np 
import cPickle as pickle 
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
                

    print np.sum(match, axis = 0), len(data1)
    return

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
       #data_list = feature_dictionary[key][6]
       #print key, len(data_list)
       data[int(key),:] = data_list

    return data


def matcher_flann(desc1,desc2):
   # FLANN parameters
   FLANN_INDEX_KDTREE = 0
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks=50)   # or pass empty dictionary

   matcher = cv2.FlannBasedMatcher(index_params,search_params)

   matches=matcher.knnMatch(np.asarray(desc1,np.float32),np.asarray(desc2,np.float32), k=2)

   # Need to draw only good matches, so create a mask
   matchesMask = [[0,0] for i in xrange(len(matches))]

   #print len(matchesMask), " : ",
   # ratio test as per Lowe's paper
   vector = list()
   kp_info = list()
   for threshold in np.arange(0.5,0.8,.05):
      threshold_list = list()
      for i,(m,n) in enumerate(matches):
         if m.distance < threshold*n.distance:
            matchesMask[i]=[1,0]
      #print np.sum(matchesMask, axis = 0)[0], 
      threshold_list = [index for index,x in enumerate(matchesMask) if x[0] ==1]
      vector.append(np.sum(matchesMask, axis = 0)[0])
      kp_info.append(threshold_list)
   #print
   return len(matchesMask), vector, kp_info


def descriptor_matcher(method, desc1, desc2):
   start = time.time()
   if method == "flann":
      # evaluate thresholds to determine confidence 
      length, vector, kp_info = matcher_flann(desc1,desc2)

   elif method == "confidence":
      # evaluate thresholds to determine confidence 
      start = time.time()
      confidence_list = np.arange(1,6) * 10
      KNNmatch(desc1, desc2, confidence_list,0)

   end = time.time()
   return length, vector, end-start, kp_info


def process_compare(keypoint_file1, keypoint_file2):
   # Get the keypoints/descriptor files
   dictionary1 = read_json_keypoints(keypoint_file1)
   dictionary2 = read_json_keypoints(keypoint_file2)

   # Example printing dictionary key 0 
   # key is (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[index])
   #print dictionary1["0"]
   point_pt, point_size, point_angle, point_response, point_octave, point_class_id, data_list = dictionary1["0"]
   descriptor_length = len(data_list)

   # need to convert (descriptor) from dictionary1 and dictionary2 to numpy data arrays.
   desc1 = convert_feature_dictionary_to_data_array(dictionary1, descriptor_length)
   desc2 = convert_feature_dictionary_to_data_array(dictionary2, descriptor_length)

   #print desc1.shape, desc2.shape
   length, vector, flann_time, kp_info = descriptor_matcher("flann", desc1,desc2)
   #conf_time = descriptor_matcher("confidence", desc1,desc2)
   #print flann_time, conf_time
   #print keypoint_file1, keypoint_file2, flann_time
   #print "Kp_info", kp_info
   return length, vector, flann_time, kp_info


def compare_all(file_prefix, summary_file="summary.json"):
   files = glob.glob(file_prefix + '*.json')
   compare_list = list(itertools.product(files, files))

   #print "compare list", files
   comparisons = dict((index,(a,b,process_compare(a,b))) for index, (a,b) in enumerate(compare_list) if (a!=b))

   #print comparisons
   #print comparisons
   with open(summary_file, "w") as fo:
      fo.write(json.dumps(comparisons, cls=NumpyAwareJSONEncoder) + "\n")


def parse_args():
    
   parser = argparse.ArgumentParser(description='matcher')
   parser.add_argument('file_prefix', help="Need a file prefix")
   parser.add_argument('summary_file', help="Need a summary file name")

   args = parser.parse_args()
   file_prefix = str(args.file_prefix)
   summary_file = str(args.summary_file)

   compare_all(file_prefix, summary_file)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
   parse_args()


