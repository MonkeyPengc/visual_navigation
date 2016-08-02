
# brief : generates image keypoints via different hessian threshold values

# -----------------------------------------------------------------------------
# import modules

import argparse
import os
import sys

def main():
    
   # ----- parse arguments -----
   
   parser = argparse.ArgumentParser(description='Parsing')
   parser.add_argument('target', help="requires directory")
   parser.add_argument('file_type', help="requires image type: png, jpg, etc")

   args = parser.parse_args()
   target = args.target
   file_type = args.file_type

   # ----- make a directory -----
    
   if not os.path.exists(target):
      print "Target directory does not exist"
      sys.exit()

   os.chdir(target)
   cwd = os.getcwd()
   cmd = "python " + os.path.join(os.environ['VISUAL_NAV_PROJECT'],"generate_all_image_keypoints.py")

   # ----- set thresholds -----
   for min_hessian in [400,800,1600,3200]:
      os.system(cmd + " " + cwd + " " + str(file_type) + " keypoint" + str(min_hessian) + " " + str(min_hessian))
   os.chdir("../")

# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()
