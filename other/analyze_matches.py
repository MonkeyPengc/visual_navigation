
# brief : generates matching scores heatmaps using the summary json file by keypoints files self-matching

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
from operator import itemgetter

# import order is important
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# implementation methods 

def find_max_grid_dimensions(sum_list):
   grid_list = [os.path.splitext(f1)[0] for (f1,f2,count,scores) in sum_list]
   grid = [(f1.split("_")[-2],f1.split("_")[-1]) for f1 in grid_list]
   new_grid = [(int(a),int(b)) for (a,b) in grid]
   rows =  max(new_grid,key=itemgetter(0))[0]
   cols =  max(new_grid,key=itemgetter(1))[1]
   return int(rows)+1, int(cols)+1


def analyze_summary_file(summary_file):
   # get the keypoints/descriptor files

   with open(summary_file, "r") as fi:
      sum_dict = json.load(fi)

   sum_list = [(sum_dict[k][0], sum_dict[k][1],sum_dict[k][2][0], sum_dict[k][2][1]) for k in sum_dict]

   # Find the X and Y dimensions
   rows,cols = find_max_grid_dimensions(sum_list)
   data_grid  = np.zeros((int(rows),int(cols)))
   count_grid  = np.zeros((int(rows),int(cols)))
   threshold_position = 4
   for f1,f2,count,vector in sum_list:
      name = (os.path.splitext(f1)[0]).split("_")
      grid_x = name[-1]
      grid_y = name[-2]
      score = vector[threshold_position]
      data_grid [grid_y,grid_x] += score
      count_grid [grid_y,grid_x] = count
      #print grid_y,grid_x, grid_data[grid_y,grid_x]

   avg_data_grid = data_grid / (rows*cols - 1)
   print rows, cols

   scale_data_grid = avg_data_grid / count_grid
   return avg_data_grid, scale_data_grid


def config_heatmap(grid_data, ax, row_labels, column_labels):
   ## heatmap axis configurations   

   #put the major ticks at the middle of each cell
   ax.set_xticks(np.arange(grid_data.shape[0])+0.5, minor=False)
   ax.set_yticks(np.arange(grid_data.shape[1])+0.5, minor=False)

   # want a more natural, table-like display
   ax.invert_yaxis()
   ax.xaxis.tick_top()

   ax.set_xticklabels(row_labels, minor=False)
   ax.set_yticklabels(column_labels, minor=False)    

   return ax


def generate_heatmap(avg_data, scale_data):   

   dim = len(avg_data)
   row_labels = list(range(0, dim))
   column_labels = list(range(0, dim))

   fig, ax = plt.subplots()
   pl = plt.pcolor(avg_data,cmap=plt.cm.Blues)
   plt.title('average keypoint matches', y=1.08)
   cbar = plt.colorbar(pl)
   ax = config_heatmap(avg_data, ax, row_labels, column_labels)   
   plt.savefig('avg_data_grid_heatmap')

   fig, ax = plt.subplots()
   pl = plt.pcolor(scale_data,cmap=plt.cm.Blues)
   plt.title('density of average matches', y=1.08)
   cbar = plt.colorbar(pl)
   ax = config_heatmap(scale_data, ax, row_labels, column_labels)
   plt.savefig('scale_data_grid_heatmap')


def main_analyze_summary_file():
    
   parser = argparse.ArgumentParser(description='matcher')
   parser.add_argument('summary_file', help="Need a summary file")

   args = parser.parse_args()
   summary_file = str(args.summary_file)

   avg_data_grid, scale_data_grid = analyze_summary_file(summary_file)
   
   generate_heatmap(avg_data_grid, scale_data_grid)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
   main_analyze_summary_file()


