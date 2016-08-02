# script-10

# author: Cheng Peng
# brief: this script plots a 2-D graph that shows the number of matches per frame verse all the maps in database. 

import matplotlib.pyplot as plt
import argparse
import glob
import os
import json

def make_graph(root_directory):
    
    file_list = list()
    
    for dirName, subdir, fileList in os.walk(root_directory):
        for f in fileList:
            if f.endswith("summary_matches.json"):
                file_list.append(os.path.join(dirName, f))
    file_list.sort(key=os.path.getmtime)

    n_match = list()
    for m_file in file_list:
        with open(m_file, 'r') as match_dict:
            matches = json.load(match_dict)
            if "position" in matches.keys():
                n_match.append(len(matches["position"]))

    num_frame = len(n_match)
    n_frame = [i for i in range(num_frame)]
    plt.plot(n_frame, n_match, '--')
    plt.title('chatfield')
    plt.ylabel('number of matches per frame')
    plt.xlabel('flight frame')
    plt.show()


def main():
    
    parser = argparse.ArgumentParser(description='make a graph that plots the number of matches per frame over the ~100 frames')
    parser.add_argument('src_directory', help="requires a root directory contains matches data")

    args = parser.parse_args()
    src_directory = args.src_directory

    make_graph(src_directory)

# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()
