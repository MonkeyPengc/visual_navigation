# script-9

# author: Cheng Peng
# brief: this script turns all the heatmaps into animation (a .mp4 file).

import moviepy.editor as mpy
import argparse
import glob
import os


def make_gif(root_directory):
    
    image_list = list()
    
    for dirName, subdir, fileList in os.walk(root_directory):
        for f in fileList:
            if f.endswith("tracker_grid_heatmap.png"):
                image_list.append(os.path.join(dirName, f))
    image_list.sort(key=os.path.getmtime)
    clip = mpy.ImageSequenceClip(image_list, fps=1)
    clip.write_videofile("tracker.mp4",fps=1)


def main():
    
    parser = argparse.ArgumentParser(description='make matching animation')
    parser.add_argument('src_directory', help="requires a root directory contains matched images")

    args = parser.parse_args()
    src_directory = args.src_directory

    make_gif(src_directory)

# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()
