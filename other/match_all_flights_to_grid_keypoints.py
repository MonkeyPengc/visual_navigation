
# brief : match flights with different thresholds with the database 
# export VISUAL_NAV_PROJECT={the scripts location}

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


def parse_args():
   parser = argparse.ArgumentParser(description='matcher')
   parser.add_argument('file_prefix', help="Need a file prefix")

   args = parser.parse_args()
   prefix = str(args.file_prefix)

   cmd = "python " + os.path.join(os.environ['VISUAL_NAV_PROJECT'],"match_flight_to_grid_keypoints.py")
  
   location_db = "/home/danconnors/GoogleMapGridGrab/Location"
   list_of_dir = glob.glob(prefix+'*')
   sorted_list = sorted(list_of_dir)
   for dir in sorted_list:
      os.chdir(dir)
      for parm in [400,800,1600,3200]:
          run_cmd = cmd + " " + "flight" + str(parm) + "/flight.json" + " gridDict " + os.path.join(location_db,"gridDict"+str(parm)+"/")
          #print run_cmd
          os.system(run_cmd)
      os.chdir("../")

if __name__ == '__main__':
   parse_args()


