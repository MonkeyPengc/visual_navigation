# script-1

# author: Cheng Peng
# brief : given a location(latitude and longitude), generates database grid images and a .json (corner/center gps)
# image source: Google Map API

# -----------------------------------------------------------------------------
# import modules

import math
import argparse
import numpy as np
import json
import os
import time

# -----------------------------------------------------------------------------
# implementation methods

tileSize = 256
initialResolution = 2 * math.pi * 6378137 / tileSize
# 156543.03392804062 for tileSize 256 pixels
originShift = 2 * math.pi * 6378137 / 2.0
# 20037508.342789244


def LatLonToMeters(lat, lon):
    # Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913

    mx = lon * originShift / 180.0
    my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

    my = my * originShift / 180.0
    return mx, my


def MetersToPixels(mx, my, zoom):
    # Converts EPSG:900913 to pyramid pixel coordinates in given zoom level

    res = initialResolution / (2**zoom)
    px = (mx + originShift) / res
    py = (my + originShift) / res
    return px, py
    

def PixelsToMeters(px, py, zoom):
    # Converts pixel coordinates in given zoom level of pyramid to EPSG:900913

    res = initialResolution / (2**zoom)
    mx = px * res - originShift
    my = py * res - originShift
    return mx, my


def MetersToLatLon(mx, my):
    # Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum

    lon = (mx / originShift) * 180.0
    lat = (my / originShift) * 180.0

    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon


def GpsCorners(lat, lon, imgsize, zoom):
    ## Calculate GPS pairs of values for the corners of the square image
    
    mx, my = LatLonToMeters(lat, lon)
    pixel_x, pixel_y = MetersToPixels(mx, my, zoom)

    ## corner pixel coordinate 
    NW = pixel_x - imgsize/2, pixel_y + imgsize/2
    NE = pixel_x + imgsize/2, pixel_y + imgsize/2
    SW = pixel_x - imgsize/2, pixel_y - imgsize/2
    SE = pixel_x + imgsize/2, pixel_y - imgsize/2
    gps = (NW, NE, SW, SE)
    
    list_of_corners = []
    
    for corner in gps:
        mx, my = PixelsToMeters(corner[0], corner[1], zoom)
        cx, cy = MetersToLatLon(mx, my)
        list_of_corners.append((cx, cy))
    
    return list_of_corners


def ApproximateLatLong(x, y, list_of_corners, imagesize, zoom):
    ## Approximate lat and long of a point in the map 

    delta_x = 0.5 - x
    delta_y = 0.5 - y
    pixel_x = imagesize/2 + delta_x * imagesize
    pixel_y = imagesize/2 + delta_y * imagesize
    
    pixel_nw = (0, 0)
    pixel_ne = (imagesize, 0)
    pixel_sw = (0, imagesize)
    pixel_se = (imagesize, imagesize)
    
    col_scale = (pixel_x - pixel_nw[0]) / imagesize
    row_scale = (pixel_y - pixel_nw[1]) / imagesize
    
    nw = list_of_corners[0]
    ne = list_of_corners[1]
    sw = list_of_corners[2]
    se = list_of_corners[3]
    
    lat = ne[0] + row_scale * (se[0] - ne[0])
    lon = nw[1] + col_scale * (ne[1] - nw[1]) 
    
    return lat, lon


def GenerateBlockCenter(gs, list_of_corners, imgsize, zoom):
    ## Generate gps location for each grid frame image 
    
    grid_size = float(gs)
    block_size = 1/grid_size
    center_to_right = 1/2 * block_size  
    center_to_bottom = 1/2 * block_size
    frame = int(gs) - 1
    centers = dict()
    
    for down in np.arange(center_to_bottom, 1, block_size):
        index = str(frame)
        for right in np.arange(center_to_right, 1, block_size):
            appr_lat, appr_lon = ApproximateLatLong(right, down, list_of_corners, imagesize=imgsize, zoom=zoom)
            if index not in centers.keys():
                centers[index] = []
            centers[index].append((appr_lat, appr_lon))
        frame = frame - 1

    return centers


def GenerateGrid(map_name, centers, gs, imgsize, zoom):
    ## Generate corners and center of a grid map
    
    frame_size = 1 / int(gs) * imgsize
    framedictionary = dict()
    for grid_y in centers.keys():
        grid_x = int(gs)
        for block_lat, block_lon in centers[grid_y]:
            grid_x = grid_x - 1
            index = str(map_name) + "_z" + str(zoom) + "_" + str(grid_x) + "_" + grid_y
            if index not in framedictionary.keys():
                framedictionary[index] = []
            block_corners = GpsCorners(block_lat, block_lon, frame_size, zoom)
            block_center = [block_lat, block_lon] # added
            framedictionary[index] = [block_corners, block_center]
                  
    return framedictionary


def main():
    
    # ----- parse arguments -----
    
    parser = argparse.ArgumentParser(description='Parsing')
    parser.add_argument('lat', help="Need a latitude as input")
    parser.add_argument('lon', help="Need a longitude as input")
    parser.add_argument('grid', help="Need a grid size as input")  # grid size
    parser.add_argument('fileout', help="Need a filename as output")
    parser.add_argument('zoom', default=19, nargs='?')  # zoom parameter, change it based on the lens distance
    parser.add_argument('imagesize', default=10240, nargs='?')  # orginal map size
    args = parser.parse_args()
    
    lat = float(args.lat)
    lon = float(args.lon)
    imgsize = args.imagesize
    zoom = args.zoom
    file_out = args.fileout

    # ----- make a directory -----
    
    if not os.path.exists(file_out):
       os.makedirs(file_out)
    os.chdir(file_out)
    
    list_of_corners = GpsCorners(lat, lon, imgsize, zoom)
    centers = GenerateBlockCenter(args.grid, list_of_corners, imgsize, zoom)
    grids = GenerateGrid(file_out, centers, args.grid, imgsize, zoom)
    
    ## write corners/center to a json file
    file_o = "{0}.json".format(file_out)
    with open(file_o, "w") as fo:
        fo.write(json.dumps(grids) + "\n")
        
    ## request all maps
    size = (imgsize/int(args.grid), imgsize/int(args.grid))  # 512x512
    maptype = "satellite"
    imgformat = "png"
    scale = 2

    with open(file_o, "r") as fi:    
        data = json.load(fi)
        for key in data.keys():
            name, zoomLevel, rowIndex, colIndex = key.split("_")   # key string: "Map_Z16_6_15"
            lat, lon =  list(data[key][1])  ## get center lat/long
            req = "http://maps.google.com/maps/api/staticmap?"
            req += "center=%f,%f&" % (lat, lon)
            req += "zoom=%i&" % zoom
            req += "size=%ix%i&" % (size)
            req += "format=%s&" % imgformat
            req += "maptype=%s&" % maptype
            req += "scale=%i" % scale  ## resolution 2048x2048
            
            image_output_name = file_out + "_" + zoomLevel + "_" + rowIndex + "_" + colIndex + "." + imgformat
            mycmd = "curl \"" + str(req) + "\" -o " + image_output_name
            os.system(mycmd)
            time.sleep(4)  ## avoid requests coming in too fast


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()  

