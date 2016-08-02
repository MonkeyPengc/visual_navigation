# script-0

# author: Cheng Peng
# brief : turns an Video into multiple labeled images(.PNG), or into multiple labeled videos
# pre-require: moviepy package

# -----------------------------------------------------------------------------
# import modules

import argparse
from moviepy.editor import *
import cv2

# -----------------------------------------------------------------------------
# methods implementation

def calculate_stamp_second(time_stamp):
    ## have the time stamp convert to second
    
    h, m, s = time_stamp.split(':')
    det_h = int(h)
    det_m = int(m)
    det_s = int(s)
    time_sec = det_h * 3600 + det_m * 60 + det_s   

    return time_sec


def generate_time_sequence(video, time_start, step, time_end):
    ## generate a list of time sequences  
    
    time_queue = list() 
    duration = video.duration
    start_stamp_sec = calculate_stamp_second(time_start)
    end_stamp_sec = calculate_stamp_second(time_end)
    
    while start_stamp_sec <= duration and start_stamp_sec <= end_stamp_sec:
        
        time_queue.append(time_start) # add the initial time stamp
        step_sec = calculate_stamp_second(step) 
        start_stamp_sec = start_stamp_sec + step_sec  # get the next stamp in second
        
        new_second = start_stamp_sec % 60
        new_minute = int(start_stamp_sec / 60)
        if new_minute >= 60:
            new_hour = int(new_minute / 60) 
            new_minute = new_minute % 60
        else:
            new_hour = 0
        
        # update the time stamp string
        time_start = str(new_hour) + ":" + str(new_minute) + ":" + str(new_second)
            
    return time_queue
    

def video_to_frame(video, time_queue):
    ## turn input video into multiple images
    
    index = 0
    for frame_time in time_queue:
        label = "image_" + str(index) + "_" + frame_time + ".png"
        video.save_frame(label, t=frame_time)
        index += 1

        
def video_to_video(video, time_queue):
    ## turn input video into multiple videos
    
    for index in range(len(time_queue)-1):
        start_stamp = time_queue[index]
        end_stamp = time_queue[index+1]
        start_sec = calculate_stamp_second(start_stamp)
        end_sec = calculate_stamp_second(end_stamp)
        clip = video.subclip(start_sec, end_sec)
        video_label = "video_" + str(index) + "_" + start_stamp + "-" + end_stamp + ".mp4"
        clip.write_videofile(video_label)
        
        
def main():
    
    # ----- parse arguments -----
    
    parser = argparse.ArgumentParser(description='Convert video file to multiple frames/videos')
    parser.add_argument('video_file', help="Need a video file as input")
    parser.add_argument('mode', help="Need a mode either 'frame' or 'video' to parse")
    parser.add_argument('start', help="Need a start time point(hour:minute:second) to start")
    parser.add_argument('step', help="Need a step size(hour:minute:second)")
    parser.add_argument('end', help="Need a end time point(hour:minute:second) to terminate")
    args = parser.parse_args()
    
    mode = args.mode
    file_in = args.video_file
    start = args.start
    step = args.step
    end = args.end
    
    
    # ----- open the video -----
    
    cap = cv2.VideoCapture(file_in)
        
    if not(cap.isOpened()):
        print("Failed to open video")
        exit(1)
    
    
    # ----- make sure the input time is correct -----
    
    vid = VideoFileClip(file_in)
    vid_time = vid.duration
    st_time = calculate_stamp_second(start)
    ed_time = calculate_stamp_second(end)

    if st_time >= 0 and st_time < ed_time and ed_time <= vid_time:
    
        # ----- convert options -----
    
        if mode == 'frame':
            timeQueue = generate_time_sequence(vid, start, step, end)
            video_to_frame(vid, timeQueue)
    
        if mode == 'video':
            timeQueue = generate_time_sequence(vid, start, step, end)
            video_to_video(vid, timeQueue)

    else:
        print("Input time is not correct")
        exit(1)

# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()  
