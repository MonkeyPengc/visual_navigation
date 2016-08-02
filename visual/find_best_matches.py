# script-6

# author: Cheng Peng
# brief : this outputs 1) a matching score heatmap (one flight frame vs multiple database maps), 
# 2) a plot of matched keypoints for the best matched database image with the flight image, 
# 3) a plot of the smallest bounding box of the matching area.

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
from operator import itemgetter
from collections import defaultdict
from collections import Counter
from match_flight_to_grid_keypoints import NumpyAwareJSONEncoder

# import order is important
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# implementation methods


def find_max_grid_dimensions(sum_list):
   grid_list = [os.path.splitext(f2)[0] for (f1,f2,count,index,score) in sum_list]
   grid = [(f2.split("_")[-2],f2.split("_")[-1]) for f2 in grid_list]
   new_grid = [(int(a),int(b)) for (a,b) in grid]
   rows =  max(new_grid,key=itemgetter(0))[0]
   cols =  max(new_grid,key=itemgetter(1))[1]

   return int(rows)+1, int(cols)+1


def analyze_summary_file(summary_file, ratio_position):
   ## generates count and scores of the matches for grids

   dim = 16  # default dimension of database grids 16x16
   
   with open(summary_file, "r") as fi:
      sum_dict = json.load(fi)

   sum_list = [(sum_dict[k][0], sum_dict[k][1], sum_dict[k][2][1], sum_dict[k][2][3], sum_dict[k][2][4]) for k in sum_dict]

   # find the X and Y dimensions
   rows, cols = find_max_grid_dimensions(sum_list)

   if rows < dim or cols < dim:
      data_grid  = np.zeros((dim, dim))  ## array that stores the number of matches for drawing heatmap

   else:
      data_grid  = np.zeros((int(rows), int(cols)))

   index_grid = defaultdict(list) ## dict that stores the index of matches
   score_grid = defaultdict(list) ## dict that stores the score of matches

   for f1, f2, count, index, score in sum_list:
      name = (os.path.splitext(f2)[0]).split("_")
      grid_x = int(name[-2])
      grid_y = int(name[-1])
      number = count[ratio_position]
      data_grid[grid_y, grid_x] = number   ## comment this line to run process_flight.py
      key = str(grid_x) + ',' + str(grid_y)
      index_grid[key] = index[ratio_position]
      score_grid[key] = score[ratio_position]

   return data_grid, index_grid, score_grid

   
def config_heatmap(data_grid, ax, row_labels, column_labels):
   ## heatmap axis configurations   

   #put the major ticks at the middle of each cell
   ax.set_xticks(np.arange(data_grid.shape[0])+0.5, minor=False)
   ax.set_yticks(np.arange(data_grid.shape[1])+0.5, minor=False)

   #want a more natural, table-like display
   ax.invert_yaxis()
   ax.xaxis.tick_top()
   ax.set_xticklabels(row_labels, minor=False)
   ax.set_yticklabels(column_labels, minor=False)    

   return ax


def generate_heatmap(data_grid, dst_directory):
   ## generates a heat map for the number of matches

   dim = len(data_grid)
   row_labels = list(range(0, dim))
   column_labels = list(range(0, dim))

   fig, ax = plt.subplots()
   pl = plt.pcolor(data_grid, cmap=plt.cm.Blues)
   plt.title('keypoint matches with database', y=1.08)
   cbar = plt.colorbar(pl)
   ax = config_heatmap(data_grid, ax, row_labels, column_labels)
   full_path = os.path.join(dst_directory, 'data_grid_heatmap')
   plt.savefig(full_path)


def boundary_plot(keypoint_position, match_img, dst_directory):
   ## plot the keypoints bounding box on the best matched grid

   img = cv2.imread(match_img)
   x_list = list()
   y_list = list()
   for x,y in keypoint_position:
       x_list.append(x)
       y_list.append(y)
   sort_x = sorted(x_list)    
   sort_y = sorted(y_list)

   try:
       lefttop_x = min(sort_x)
       rightbottom_x = max(sort_x)
       lefttop_y = min(sort_y)
       rightbottom_y = max(sort_y)
   except ValueError:
       pass

   else:
       plt.figure(figsize=(10,10))
       cv2.rectangle(img, (int(lefttop_x), int(lefttop_y)), (int(rightbottom_x), int(rightbottom_y)), (255,0,0), 2)
       plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
       full_path = os.path.join(dst_directory, 'keypoints_boundary')
       plt.savefig(full_path)


def flann_match(query_image, train_image, threshold, ratio_position):

   query = cv2.imread(query_image, 0)  
   train = cv2.imread(train_image, 0)
   surf = cv2.xfeatures2d.SURF_create()
   surf.setHessianThreshold(threshold)
   kp1, desc1 = surf.detectAndCompute(query, None)
   kp2, desc2 = surf.detectAndCompute(train, None)
   FLANN_INDEX_KDTREE = 0
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks = 50)
    
   flann = cv2.FlannBasedMatcher(index_params, search_params)
   matches = flann.knnMatch(desc1, desc2, k=2)
   # store all the good matches as per Lowe's ratio test.
   good = []
   threshold_list = [i for i in np.arange(0.5, 0.8, .05)]
   for m,n in matches:
       if m.distance < threshold_list[ratio_position]*n.distance:
           good.append(m)

   return kp1, kp2, good 


def search_gps_data(pixel_position, gps_file):
   ## extract matched keypoint pixel coordinates and gps

   gps_list = list()
  # print(pixel_position)
   with open(gps_file, "r") as fi:    
       data = json.load(fi)
       for pixel_pair in pixel_position:
           for key in data.keys():
               #print(data[key][0][0])
               if data[key][0][0][0] == pixel_pair[0] and data[key][0][0][1] == pixel_pair[1]:
                    gps_list.append(data[key][1])

   return gps_list

 
def find_grid_gps(max_grid, query_image, directory, db_directory, threshold, ratio_position=3):
   ## extract matched keypoint GPS in the best matched grid

   db_images =  glob.glob(os.path.join(db_directory,'*.png'))
   gps_files = glob.glob(os.path.join(directory,'*.json'))
   grid_x = max_grid[0]
   grid_y = max_grid[1]

   for train_image in db_images:
       name = os.path.basename(train_image)
       grid = os.path.splitext(name)[0]
       index = grid.split('_')
       index_y = int(index[-1])
       index_x = int(index[-2])
        
       if index_x == grid_x and index_y == grid_y:
           kp1, kp2, matches = flann_match(query_image, train_image, threshold, ratio_position)
           query_position = [kp1[m.queryIdx].pt for m in matches]
           train_position = [kp2[m.trainIdx].pt for m in matches]
           
           for match_gpsFile in gps_files:
                gps_grid = os.path.splitext(match_gpsFile)[0]
                gps_index = gps_grid.split('_')
                gps_y = int(gps_index[-1])
                gps_x = int(gps_index[-2])
                if gps_x == grid_x and gps_y == grid_y:
                    #print(match_gpsFile)
                    keypoint_gpsPosition = search_gps_data(train_position, match_gpsFile)  ## find gps for matches

           return keypoint_gpsPosition, query_position, train_position, train_image


def draw_top_kimage(query_image, train_image, dst_directory, threshold, ratio_position):
    ## generates matching images of top k matched grids with the query frame

    cmd = "python " + os.path.join(os.environ['VISUAL_NAV_PROJECT'],"match_image.py")
    cmd += " " + query_image + " " + train_image + " " + dst_directory + " " + threshold + " " + str(ratio_position)
    #print(cmd)
    os.system(cmd)

def write_summary(data, dst_directory, summary_file="summary_matches.json"):

    with open(os.path.join(dst_directory, summary_file), "w") as fo:
        fo.write(json.dumps(data, cls=NumpyAwareJSONEncoder) + "\n")


def filter_grid(query_image, train_image, threshold, ratio_position, repeated_points):
    ## filter out grid candidates that has repeated matches
    
    keypoints_query, keypoints_train, matches = flann_match(query_image, train_image, threshold, ratio_position)
    kp_position = [keypoints_train[m.trainIdx].pt for m in matches]
    
    if not kp_position == []:
        repeat = Counter(kp_position).most_common(1)[0][1]
    
        if repeat > repeated_points:  ## min_repeat threshold
            return 0
        else:
            return 1
    else:  # in case of 0 matches
        return 0

def find_top_kscore(index_grid, score_grid, data_grid, query_image, db_directory, dst_directory, threshold, ratio_position=2, k=8, repeated_points=5):
    ## generates the best matched grid and prints number, index, and score of top k matched grids

    y = data_grid.shape[1]
    index_flat = list(np.argsort(data_grid, axis=None)[-k:])
    db_images =  glob.glob(os.path.join(db_directory,'*.png'))
    neg_candidate = list()
    count_number = defaultdict(int)
    count_distance = defaultdict(list)
    matches_dict = defaultdict(list)
    
    for i, index in enumerate(index_flat):
        col = index % y
        row = index // y
        pattern = str(col) + ',' + str(row)
        if pattern in index_grid.keys():
            print([col, row], len(index_grid[pattern]), index_grid[pattern], score_grid[pattern])
            # writes the infomation of top k matched grid into a dictionary
            matches_dict[i] = [col, row], len(index_grid[pattern]), index_grid[pattern], score_grid[pattern]
            for top_grid in db_images:
                grid = os.path.splitext(top_grid)[0]
                name = grid.split('_')
                index_y = int(name[-1])
                index_x = int(name[-2])
                if col == index_x and row == index_y:
                    #draw_top_kimage(query_image, top_grid, dst_directory, threshold, ratio_position) ## call to draw the matches
                    flag = filter_grid(query_image, top_grid, int(threshold), ratio_position, repeated_points)
                    if flag == 0:
                        neg_candidate.append(index)  # add the bad candidate
    
    top_list = [item for item in index_flat if not item in neg_candidate]  # generate a better candidates list
    
    if not top_list == []:
        for item in top_list:
            col = item % y
            row = item // y
            count_number[item] = data_grid[row, col]
            #count_distance[item] = sum(score_grid[str(col)+','+str(row)]) / len(score_grid[str(col)+','+str(row)])
            print("better candidate:", [col, row])
    
        top_flat_number = max(count_number.items(), key=itemgetter(1))[0]  # find the grid that has max number of matches
        #top_flat_distance = min(count_distance.items(), key=itemgetter(1))[0] # find the grid that has min avg distance of matches
        
        top_col = top_flat_number % y
        top_row = top_flat_number // y
        top = [top_col, top_row]  # return the best matched position

        return top, matches_dict

    return None, matches_dict

def main_analyze_summary_file():
    
   # ----- parse arguments -----
   
   parser = argparse.ArgumentParser(description='matcher')
   parser.add_argument('summary_file', help="requires a summary file")
   parser.add_argument('kp_gps_directory', help="requires a keypoint-GPS directory")
   parser.add_argument('query_image', help="requires the query image")
   parser.add_argument('images_directory', help="requires the grid database directory")
   parser.add_argument('dst_directory', help="requires a directory for all the output")
   parser.add_argument('hessian_threshold', help="requires the hessian threshold")
   parser.add_argument('ratio_position', help="requires the position of ratio value [0-6]") # 2 = ratio 0.65 by default
   parser.add_argument('top_candidates', help="requires the number of top matched grids") # 8 by default
   parser.add_argument('repeated_points', help="requires the number of repeated matches") # 5 by default
   args = parser.parse_args()
   
   summary_file = args.summary_file
   directory = args.kp_gps_directory 
   query_image = args.query_image
   db_directory = args.images_directory
   dst_directory = args.dst_directory
   threshold = int(args.hessian_threshold)
   ratio_position = int(args.ratio_position)
   top_k = int(args.top_candidates)
   repeated_points = int(args.repeated_points)
   
   
   # ----- make a directory -----
   
   if not os.path.exists(dst_directory):
       os.makedirs(dst_directory)
   
   
   # ----- call components -----
   
   data_grid, index_grid, score_grid = analyze_summary_file(summary_file, ratio_position)

   generate_heatmap(data_grid, dst_directory)
   
   best_grid, matches_data = find_top_kscore(index_grid, score_grid, data_grid, query_image, db_directory, dst_directory, args.hessian_threshold, ratio_position, top_k, repeated_points)

   if best_grid != None:
      gpsPosition, pixel_position1, pixel_position2, match_img = find_grid_gps(best_grid, query_image, directory, db_directory, threshold, ratio_position)
   
      boundary_plot(pixel_position2, match_img, dst_directory)
      
      # add the best matched grid to the dictionary
      matches_data["-1"] = best_grid
      
      # add the pixel positions of matches to the dictionary
      matches_data["position"] = pixel_position2
      write_summary(matches_data, dst_directory)

      print("Best matched grid:")
      print(best_grid)
      #print("pixel coordinates of matched keypoints in query image:")
      #print(pixel_position1)
      #print("pixel coordinates of matched keypoints in train image:")
      #print(pixel_position2)
      #print("gps of matched keypoints:")
      #print(gpsPosition)

   else:
      
      matches_data["-1"] = [None, None]
      matches_data["position"] = [None]
      write_summary(matches_data, dst_directory)
      print("No match grid detected.")


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
   main_analyze_summary_file()



