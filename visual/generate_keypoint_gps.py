# script-4

# author: Cheng Peng
# brief : generates the gps of all keypoints for each image in the grid database, where gps data are stored in .json files 

# -----------------------------------------------------------------------------
# import modules

import glob
import json
import os
import argparse


# -----------------------------------------------------------------------------
# implementation methods

def ApproximateLatLong(pixel_coordinates, map_corners, imagesize, zoom):
    ## calculate approximate lat and long of a key point in the map 

    pixel_x = pixel_coordinates[0][1]  ## double check
    pixel_y = pixel_coordinates[0][0]
    
    pixel_nw = (0, 0)
    col_scale = (pixel_x - pixel_nw[0]) / imagesize
    row_scale = (pixel_y - pixel_nw[1]) / imagesize
    
    nw = map_corners[0]
    ne = map_corners[1]
    sw = map_corners[2]
    se = map_corners[3]
    
    lat = ne[0] + row_scale * (se[0] - ne[0])
    lon = nw[1] + col_scale * (ne[1] - nw[1]) 
    
    return lat, lon      


def match_grid(kp_gridx, kp_gridy, imgsize, zoom, gps_file, grid_keypoints):
    ## match a keypoints json file with map grid, calculate gps for each keypoint in that file
    
    with open(gps_file, 'r') as fi:
        gps_data = json.load(fi)
        for key in gps_data.keys():   # key = "grid_z17_x_y"
            map_index = key.split('_')
            map_y = map_index[-1]
            map_x = map_index[-2]
           # print(map_x, map_y, kp_gridx, kp_gridy)            
            if map_x == kp_gridx and map_y == kp_gridy:
               # print("get here")
                position_gps = dict()
                map_corners = gps_data[key][0]  # get the gps of corners
                for index in grid_keypoints.keys():
                    kp_position = grid_keypoints[index]  # search each keypoint's position
                    kp_lat, kp_lon = ApproximateLatLong(kp_position, map_corners, imgsize, zoom)  # calculation
                    if not index in position_gps.keys():
                        position_gps[index] = []
                    position_gps[index] = kp_position, [kp_lat, kp_lon]
                
    return position_gps
          
        
def write_keypoint_gps(keypoint_dir, imgsize, zoom, gps_file, keypoint_gps_dir):
    ## read all the keypoint json files, extract positions, calculate gps, write to new json files
    
    files = glob.glob(os.path.join(keypoint_dir,'*.json')) 

    for kp_file in files:
        kp = os.path.splitext(kp_file)[0]  # "grid_z17_x_y.json"
        kp_in_grid = kp.split('_')
        kp_gridy = kp_in_grid[-1]
        kp_gridx = kp_in_grid[-2]
           
        grid_keypoints = dict()
        with open(kp_file, 'r') as ki:
            kp_data = json.load(ki)
            for key in kp_data.keys():
                kp_coordinates = kp_data[key][0]  # get the keypoint position
                if not key in grid_keypoints.keys():
                    grid_keypoints[key] = []
                grid_keypoints[key].append(kp_coordinates)
        #print(grid_keypoints)
        kp_gps_dict = match_grid(kp_gridx, kp_gridy, imgsize, zoom, gps_file, grid_keypoints)
        #print(kp_gps_dict)
        new_filename = "keypoint_position_gps_grid_" + kp_gridx + "_" + kp_gridy + ".json"
        with open(os.path.join(keypoint_gps_dir, new_filename), "w") as fo:
            fo.write(json.dumps(kp_gps_dict) + "\n")
               
            
def main():

    # ----- parse arguments -----
    
    parser = argparse.ArgumentParser(description='calculate keypoint gps')
    parser.add_argument('kp_directory', help="Need a keypoint files directory")
    parser.add_argument('gps_file', help="Need the summary GPS file")  ## grid.json file
    parser.add_argument('kpgps_directory', help="Need a destination directory")
    parser.add_argument('zoom', default=19, nargs='?')
    parser.add_argument('imagesize', default=640, nargs='?')
    args = parser.parse_args()

    keypoint_dir = args.kp_directory
    gps_file = args.gps_file
    image_size = int(args.imagesize)
    zoom = int(args.zoom)
    keypoint_gps_dir = args.kpgps_directory
   
    # ----- make a directory -----
    
    if not os.path.exists(keypoint_gps_dir):
       os.makedirs(keypoint_gps_dir) 


    write_keypoint_gps(keypoint_dir, image_size, zoom, gps_file, keypoint_gps_dir)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()  


