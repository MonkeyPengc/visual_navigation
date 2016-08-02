
# brief : match video frames with the satellite database
# requires to install moviepy library

# -----------------------------------------------------------------------------
# import modules


import cv2
from moviepy.editor import *
from process_video_file import calculate_stamp_second, generate_time_sequence
import generate_image_keypoints
import match_flight_to_grid_keypoints
from find_best_matches import analyze_summary_file
import argparse
import os
import shutil


# -----------------------------------------------------------------------------
# implementation


def get_flight_frame(video, index, time_frame, image_type=".png"):
## generate flight frame

    # save the frame
    label = "image_" + str(index) + "_" + str(time_frame) + image_type
    video.save_frame(label, t=time_frame)

    return label


def generate_a_subdatabase(database, db_center, dst):
## generate a sub-database directory for each frame to sub-database matching

    x, y = db_center
    
    ## get the 3x3 neighbors centered at the best matched grid
    index_list = [(x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y), (x, y), (x+1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]

    for grid_x, grid_y in index_list:
        file_name = "grid_z19_" + str(grid_x) + "_" + str(grid_y) + ".json"
        shutil.copy(os.path.join(database, file_name), dst)


def perform_matching(image_file, database, grid, index, hessian_threshold=400, feature_type="SURF", image_type=".png"):
## matching frame keypoint file with sub-database keypoint files

    # generate keypoints and descriptor of the frame
    kp, desc = generate_image_keypoints.generate_keypoint_image(image_file, hessian_threshold)
    
    # write frame keypoints into a json file
    keypoint_file = os.path.basename(image_file.replace(image_type, '.json'))
    generate_image_keypoints.write_json_keypoints(keypoint_file, kp, desc)
    
    # make a directory that stores keypoint files of the sub-database
    sub_dst = "frame_database_" + str(index)
    if not os.path.exists(sub_dst):
        os.makedirs(sub_dst)
    
    # match a frame with the sub-database
    generate_a_subdatabase(database, grid, sub_dst)
    summary_file = "summary_" + keypoint_file
    match_flight_to_grid_keypoints.compare_flight(keypoint_file, sub_dst, summary_file)
    data_grid, max_grid = analyze_summary_file(summary_file)

    return max_grid


def Process_Flight_Video(video_file, satellite_database, time_start, time_interval, time_end):
## implement the process

    video = VideoFileClip(video_file)
    timeQueue = generate_time_sequence(video, time_start, time_interval, time_end)
    
    max_grid = [8,7]  # assume that the center of grid is the video's starting point
    for index_frame, time_frame in enumerate(timeQueue):
        current_grid = max_grid
        frame = get_flight_frame(video, index_frame, time_frame)
        max_grid = perform_matching(frame, satellite_database, current_grid, index_frame)
        print(max_grid)


def main():

    # ----- parse arguments -----

    parser = argparse.ArgumentParser(description='Parsing')
    parser.add_argument('video_file', help="requires a video file")
    parser.add_argument('satellite_database', help="requires a satellite database directory")
    parser.add_argument('start', help="Need a start time point(hour:minute:second) to start")
    parser.add_argument('interval', help="Need a step size(hour:minute:second)")
    parser.add_argument('end', help="Need a end time point(hour:minute:second) to terminate")
    args = parser.parse_args()

    video_file = args.video_file
    satellite_database = args.satellite_database
    time_start = args.start
    time_interval = args.interval
    time_end = args.end

    # ----- process flight video matching -----

    Process_Flight_Video(video_file, satellite_database, time_start, time_interval, time_end)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()



