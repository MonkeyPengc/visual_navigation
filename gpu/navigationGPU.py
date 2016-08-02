
# brief: Visual Navigation implementation on GPU

# output: best match grid(including top k candidates), and a heat map of matches for each flight frame

# require: a folder that contains all the flight frames & a folder that contains database keypoint files.

# -----------------------------------------------------------------------------
# import modules

import os
import glob
import json
import shutil
import argparse
from itertools import cycle
from collections import defaultdict
from collections import Counter
from operator import itemgetter
import numpy as np
from ctypes import *
import c_cv2
import keypoints
import libng

# import order is important
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ----- define a SURF ratio tuple -----

ratio_list = (.5, .55, .6, .65, .7, .75, .8)


# ----- define static directory name -----

summary_frame_grids = "frame_grids"
summary_file = "summary.json"


# ----- define a class for .json encoder -----

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


# ----- define grids dimension -----

GRID_DIM = 16


# ----- define descriptors size -----

DESC_DIM = 64


# -----------------------------------------------------------------------------
# define functions that convert a list of kp-desc structures to c arrays

def create_c_keypoints(kpoints):
    c_kpoints = (c_cv2.c_KeyPoint*len(kpoints))()  # allocate an array
    for i in range(len(kpoints)):
        c_kpoints[i]._pt.x     = kpoints[i].pt[0]
        c_kpoints[i]._pt.y     = kpoints[i].pt[1]
        c_kpoints[i]._size     = kpoints[i].size
        c_kpoints[i]._angle    = kpoints[i].angle
        c_kpoints[i]._response = kpoints[i].response
        c_kpoints[i]._octave   = kpoints[i].octave
        c_kpoints[i]._class_id = kpoints[i].class_id
    return c_kpoints

def create_c_desc(kpoints):
    total_size = DESC_DIM*len(kpoints)
    c_desc = (c_float*total_size)()
    for x in range(len(kpoints)):
        for y in range(DESC_DIM):
            c_desc[DESC_DIM*x + y] = kpoints[x].desc[y]
    return c_desc


# -----------------------------------------------------------------------------
# define a function that collects specified keypoint files in the database

def get_files(directory, scan, best_grid=None):

    kpfile_list = list()
    dim = GRID_DIM
    
    ## get all keypoint files for scan, when a scan cycle is done or no best grid detected
    if scan == 0 or best_grid == None:
        kpfile_list = glob.glob(os.path.join(directory,'*.json'))
    
    ## otherwise, only get files of neighbour grids
    else:
        neighbour_grids = list()
        grid_x, grid_y = best_grid % dim, best_grid // dim
        # get 24 neighbours of the best match
        neighbour_grids = [(grid_x-2, grid_y-2), (grid_x-1, grid_y-2), (grid_x, grid_y-2), (grid_x+1, grid_y-2), (grid_x+2, grid_y-2), (grid_x-2, grid_y-1), (grid_x-1, grid_y-1), (grid_x, grid_y-1), (grid_x+1, grid_y-1), (grid_x+2, grid_y-1), (grid_x-2, grid_y), (grid_x-1, grid_y), (grid_x, grid_y), (grid_x+1, grid_y), (grid_x+2, grid_y), (grid_x-2, grid_y+1), (grid_x-1, grid_y+1), (grid_x, grid_y+1), (grid_x+1, grid_y+1), (grid_x+2, grid_y+1), (grid_x-2, grid_y+2), (grid_x-1, grid_y+2), (grid_x, grid_y+2), (grid_x+1, grid_y+2), (grid_x+2, grid_y+2)]
    
        for x, y in neighbour_grids:
            try:  # in case grids out of boundary
                kpfile_list.append(glob.glob(os.path.join(directory, 'grid_z19_' + str(x) + '_' + str(y) + '.*json'))[0])
            except:
                pass

    return kpfile_list


# -----------------------------------------------------------------------------
# define a function that writes matching data into a .json file

def write_summary(matching_data, location_frame, filename):
        
    with open(os.path.join(location_frame, filename), "w") as fo:
        fo.write(json.dumps(matching_data, cls=NumpyAwareJSONEncoder) + "\n")


# -----------------------------------------------------------------------------
# define functions that generate data grid array and heat map of number of matches

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

def heatmap_generator(sum_dict, location_frame, GRID_DIM):
    
    dim =  GRID_DIM  # grid size of heat map
    data_grid  = np.zeros((dim, dim))
    
    position_data = defaultdict(list)  # initialize a dictionary to store positions of matches

    sum_list = [(sum_dict[k][0], sum_dict[k][1], sum_dict[k][2], sum_dict[k][3]) for k in sum_dict]

    for f1, f2, num_match, position_vector in sum_list:
        name = (os.path.splitext(f2)[0]).split("_")
        grid_x = int(name[-2])
        grid_y = int(name[-1])
        data_grid[grid_y, grid_x] = num_match
        position_data[name[-2]+'_'+name[-1]] = position_vector

    row_labels = list(range(0, dim))
    column_labels = list(range(0, dim))
    fig, ax = plt.subplots()
    pl = plt.pcolor(data_grid, cmap=plt.cm.Blues)
    plt.title('keypoint matches with database', y=1.08)
    cbar = plt.colorbar(pl)
    ax = config_heatmap(data_grid, ax, row_labels, column_labels)

    full_path = os.path.join(location_frame, 'frame_grids_heatmap')  # name of heat map by default
    plt.savefig(full_path)

    return data_grid, position_data


# -----------------------------------------------------------------------------
# define functions that generate the best grid by sifting top k matched grids

def filter_grid(m_positions, repeated_points):
# filter out grid candidates that has repeated matches

    if not m_positions == []:
        repeat = Counter(m_positions).most_common(1)[0][1]  # get the number of repeated positions
        
        if repeat > repeated_points:  # min_repeat threshold
            return 0
        else:
            return 1
    else:  # in case of the number of match is 0
        return 0


def candidates_generator(data_grid, position_data, GRID_DIM, k=8, repeated_points=5):

    neg_candidate = list() # a list which stores index of weak candidates
    count_number = defaultdict(int)  # a dictionary which stores the number of matches of grids
    m_position = list()  # a list which stores the postion of matches
    
    index_flat = list(np.argsort(data_grid, axis=None)[-k:])
    
    dim = GRID_DIM
    for i, index in enumerate(index_flat):
        col = index % dim
        row = index // dim  # get the coordinates of candidate grid
        pattern = str(col)+'_'+str(row)
        if pattern in position_data.keys():
            flag = filter_grid(position_data[pattern], repeated_points)
            if flag == 0:
                neg_candidate.append(index)  # add to the list the index of weak candidate

    top_list = [item for item in index_flat if not item in neg_candidate]  # generate a list of better candidates
    if not top_list == []:
        for item in top_list:
            col = item % dim
            row = item // dim
            count_number[item] = data_grid[row, col]
            print("better candidate:", [col, row])
        
        grid = max(count_number.items(), key=itemgetter(1))[0]  # get the grid that has max number of matches
        x, y = grid % dim, grid // dim
        m_position = position_data[str(x)+'_'+str(y)]  # get the position of matches in the best matched grid
        return grid, m_position

    return None, None


def main():
    
    # ----- parse arguments -----
    
    parser = argparse.ArgumentParser(description='Navigation using GPU')
    parser.add_argument('flights_location', help="Requires a flight frames location")
    parser.add_argument('keypoints_db_location', help="Requires a database keypoints location")
    parser.add_argument('frames_filetype', help="Requires the file type of flight frames")
    parser.add_argument('hessian_threshold', help="Requires a hessian threshold") # suggested value: 800
    parser.add_argument('ratio_position', default=2, nargs='?', help="requires the position of ratio value [0-6]") # 2(ratio = 0.60) by default
    parser.add_argument('top_candidates', default=8, nargs='?', help="requires the number of top matched grids") # 8 by default
    parser.add_argument('repeated_points', default=5, nargs='?', help="requires the number of repeated matches") # 5 by default
    parser.add_argument('cycle_scan', default=1, nargs='?', help="requires a cycle of scan") # 1=full scan by default, 2=full/partial, 3=full/2partial

    args = parser.parse_args()
    ft_location = args.flights_location
    db_location = args.keypoints_db_location
    ft_type = args.frames_filetype
    hessian_threshold = args.hessian_threshold
    ratio_position = int(args.ratio_position)
    top_candidates = int(args.top_candidates)
    repeated_points = int(args.repeated_points)
    cycle_scan = int(args.cycle_scan)

    ratio = ratio_list[ratio_position]
    current_directory = os.getcwd()

    # -----------------------------------------------------------------------------
    # flight frames

    # ----- collect flight frames in date order -----

    list_flight_files = glob.glob(os.path.join(ft_location, '*.' + ft_type))
    list_flight_files.sort(key=os.path.getmtime)
    
    myiterator = cycle(range(cycle_scan))  # a generator that switches full scan to n partial scans
    best_grid = None
    
    for frame in list_flight_files:
        
        lib = libng.LibNg()  # initialize a class that drives GPU function

        if not(lib.LoadFrameImage(frame)):  # transfer a frame to device
            print("Could not load flight image!")
            exit(1)
        
        comparisons = defaultdict(list)  # create a dictionary to store matching data
        
        scan = next(myiterator)  # start a loop
        
        location_frame = os.path.join(current_directory, os.path.basename(frame).split('.')[0] + "_" + hessian_threshold)
        
        # ----- make a directory for the flight frame -----
        
        if not os.path.exists(location_frame):
            os.makedirs(location_frame)

        # ----- copy frame to location -----

        shutil.copy(frame, location_frame)

        
        # -----------------------------------------------------------------------------
        # process keypoints files in the database

        keypoint_file_list = get_files(db_location, scan, best_grid)

        for index, keypoint_file in enumerate(keypoint_file_list):
            
            # ----- load json file and convert it to a list of structures
            
            kpoints = keypoints.LoadJSON(keypoint_file)

            # ----- get ctypes array containing keypoint and descriptor data -----
            
            c_kpoints = create_c_keypoints(kpoints)
            c_desc = create_c_desc(kpoints)

            # ----- call external functions -----
            
            matches = c_cv2.c_Matches()  # initialize a structure that stores matches data from device
            
            lib.LoadGridKeypoints(c_kpoints)
            lib.LoadGridDescriptors(c_desc, DESC_DIM)
            lib.ConfigureSURF(int(hessian_threshold))
            lib.Process(matches, ratio)
            
            positions = (c_cv2.c_Point2f * matches.m_num)() # initialize an array that stores positions of matches
            lib.GetMatchPositions(positions)
            
            print(keypoint_file)
            
            # ----- write matching data into a dictionary -----
            
            comparisons[str(index)] = [frame, keypoint_file, matches.m_num, [(p.x, p.y) for p in positions]]
        print("****************")
        
        # ----- store data into a .json file -----
        
        #write_summary(comparisons, location_frame, summary_file)

        # ----- generate data array, postions, and heat map for matches -----

        data_grid, position_data = heatmap_generator(comparisons, location_frame, GRID_DIM)

        # ----- generate coordinates and position data of the best matched grid -----
        
        best_grid, match_position = candidates_generator(data_grid, position_data, GRID_DIM, top_candidates, repeated_points)

        print("Best matched grid:")
        print(best_grid % GRID_DIM, best_grid // GRID_DIM)
        print("Position of matches:")
        print(match_position)
        lib.__del__()


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()


