# script-3

# author: Cheng Peng
# brief : generates keypoints json files for the database images, by calling script-2
# export VISUAL_NAV_PROJECT={the scripts location}

# -----------------------------------------------------------------------------
# import modules

import cv2
import numpy as np
import json
import argparse
import os
import glob

def main():
    
   # ----- parse arguments -----
   
   parser = argparse.ArgumentParser(description='Parsing')
   parser.add_argument('image_dir', help="requires an image directory")
   parser.add_argument('image_type', help="requires an image file type")
   parser.add_argument('directory', help="requires a destination directory")
   parser.add_argument('hessian_threshold', help="requires an hessian threshold")

   args = parser.parse_args()
   image_dir = args.image_dir
   image_type = args.image_type
   hessian_threshold = args.hessian_threshold

   list_of_files = glob.glob(os.path.join(image_dir,'*.'+image_type))

   # ----- make a directory -----

   directory = str(args.directory)
   if not os.path.exists(directory):
      os.makedirs(directory)

   for file in list_of_files:
      # Feature command
      cmd = "python " + os.path.join(os.environ['VISUAL_NAV_PROJECT'],"generate_image_keypoints.py")
      cmd += " " + file + " "
      jsonfile = os.path.basename(file.replace('.'+image_type, '.json'))
      cmd += os.path.join(directory,jsonfile)
      cmd += " " + hessian_threshold
      os.system(cmd)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()
