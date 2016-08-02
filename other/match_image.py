
# -----------------------------------------------------------------------------
# import modules

import cv2
import numpy as np
#import cPickle as pickle
import json
import argparse
import os
import time
import glob
import itertools
from operator import itemgetter

# import order is important
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def flann_match(query_image, train_image, threshold, ratio_position):
    
    query = cv2.imread(query_image, 0)
    train = cv2.imread(train_image, 0)  ## best grid in database
    
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(threshold)
    kp1, desc1 = surf.detectAndCompute(query, None)
    kp2, desc2 = surf.detectAndCompute(train, None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = []
    threshold_list = [i for i in np.arange(0.5, 0.8, .05)]

    for m,n in matches:
        if m.distance < threshold_list[ratio_position]*n.distance:
            good.append(m)

    return kp1, kp2, good


def main():
    
    # ----- parse arguments -----
    
    parser = argparse.ArgumentParser(description='matcher')
    parser.add_argument('query_image', help="requires the query image")
    parser.add_argument('train_image', help="requires the train image")
    parser.add_argument('dst_directory', help="requires a directory for output image")
    parser.add_argument('hessian_threshold', help="requires the hessian threshold")
    parser.add_argument('ratio_position', help="requires the position of ratio value [0-6]") # 3 by default
    args = parser.parse_args()
    
    query_image = args.query_image
    train_image = args.train_image
    dst_directory = args.dst_directory
    threshold = int(args.hessian_threshold)
    ratio_position = int(args.ratio_position)

    keypoints_query, keypoints_train, matches = flann_match(query_image, train_image, threshold, ratio_position)
    
    # ----- make a directory -----
    
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
    

    # ----- plot the matches -----
    
    query = cv2.imread(query_image)
    train = cv2.imread(train_image)
    plt.figure(figsize=(50,50))
    image = cv2.drawMatches(query, keypoints_query, train, keypoints_train, matches, None)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    name = os.path.basename(train_image)
    grid_index = os.path.splitext(name)[0].split('_')
    grid_x = grid_index[-2]
    grid_y = grid_index[-1]
    full_path = os.path.join(dst_directory, 'matching_with_grid_' + grid_x + '_' + grid_y)
    plt.savefig(full_path)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    
    main()

