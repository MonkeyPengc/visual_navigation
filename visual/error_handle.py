# script-8

# author: Cheng Peng
# brief : this script helps to complete the glob of database maps, 
# by re-sending request for images that failed to download in the directory.

# -----------------------------------------------------------------------------
# import modules

import argparse
import os
import glob
import re
import json
import time

# -----------------------------------------------------------------------------
# implementation methods

def send_request(center, directory, image, maptype = "satellite"):

    image_info, image_format = os.path.splitext(image)
    name, zoom, rowIndex, colIndex = image_info.split('_')
    lat, lon = center
    scale = 2
    zoomLevel = int(re.split('(\D+)', zoom)[-1])
    req = "http://maps.google.com/maps/api/staticmap?"
    req += "center=%f,%f&" % (lat, lon)
    req += "zoom=%i&" % zoomLevel
    req += "size=%ix%i&" % (640,640)
    req += "format=%s&" % image_format
    req += "maptype=%s&" % maptype
    req += "scale=%i" % scale

    image_output_name = name + "_" + str(zoomLevel) + "_" + rowIndex + "_" + colIndex + "." + image_format
    output_path = os.path.join(directory, image_output_name)
    mycmd = "curl \"" + str(req) + "\" -o " + output_path  # send request
    os.system(mycmd)
    time.sleep(3)


def complete_scan(center, image_dir):
    
    image_list = glob.glob(os.path.join(image_dir,'*png'))
    
    ## scan the image directory to find any image that doesn't meet the storage minimum
    for image in image_list:
        mycmd = "identify -format '%b' " + image  # generate the storage info of an image
        storage = os.popen(mycmd).read()
        storage = str(storage)
        kb_info = re.split('(\D+)', storage)
        kb = int(kb_info[0])
        #if kb < 100:  # set storage minimum 100KB
        if kb > 2: ## set maxium
            text = image.split('/')[-1]
            image_key = os.path.splitext(text)[0]
            print(image_key)
            with open(center, "r") as fi:
                data = json.load(fi)
                for key in data.keys():
                    if key == image_key:
                        
                        send_request(data[key][1], image_dir, text)  # use the center gps of the map


def main():

    parser = argparse.ArgumentParser(description='Complete the google request')
    parser.add_argument('input_file', help="Need the center/conner file as input")
    parser.add_argument('directory', help="Need an image directory")

    args = parser.parse_args()
    center = args.input_file
    dir = args.directory
    complete_scan(center, dir)

# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()



