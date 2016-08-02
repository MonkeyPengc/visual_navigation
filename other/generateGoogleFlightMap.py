
# brief : generates the user's flight track via static google map

# -----------------------------------------------------------------------------
# import modules

import urllib
import csv
import argparse


def mark_static_google_map(mapname, center=None, zoom=16, 
                           imgsize="640x640", imgformat="png", maptype="satellite",
                           flightpath1=None, flightpath2=None, markers=None ):
    
    """ 
    example url:
    https://maps.googleapis.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=13
    &size=600x300&maptype=roadmap&markers=color:blue%7Clabel:S%7C40.702147,-74.015794
    &markers=color:green%7Clabel:G%7C40.711614,-74.012318
    &path=color:0x0000ff|weight:5|40.737102,-73.990318|40.749825,-73.987963

    """
    
    request = "http://maps.google.com/maps/api/staticmap?" # base URL, parameters seperated by &
    
    # if center and zoom  are not given, the map will show all marker locations
    if center != None:
        request += "center=%s&" % center

    
    request += "zoom=%i&" % zoom    
    request += "size=%s&" % imgsize  # tuple of ints, up to 640 by 640
    request += "format=%s&" % imgformat  # jpeg by default
    request += "maptype=%s&" % maptype  # options: roadmap, satellite, hybrid, terrain
    
    if flightpath1 != None:
        request += "&path=color:0xff0000|weight:5"    
        for location in flightpath1:
            request += "%s" % location

    if flightpath2 != None:
        request += "&path=color:0x000fff|weight:5"
        for location in flightpath2:
            request += "%s" % location
    
    # add markers (lat and lon)
    if markers != None:
        for marker in markers:
                request += "%s&" % marker
                
    # get remote data and save it to a local path
    request += "&sensor=true"
    urllib.urlretrieve(request, mapname+"."+imgformat)


def parseFlightFile(filename):
    # query location data from input file
    
    if filename == None:
      return None

    path = []
    with open(filename, 'r') as file:

        for line in file:
            if not line:
                continue
            gps = [row for row in line.split()]
            path.append("|" + '{0},{1}'.format(gps[0],gps[1]))

    return path


def main():
    
    # ----- parse arguments -----
    
    parser = argparse.ArgumentParser(description='Parsing location data file and mark data points on google map.')
    parser.add_argument('mapname', help="Need a map file name as output")
    parser.add_argument('flightfile1', help="Need a location data file as input")
    parser.add_argument('flightfile2', default=None, nargs='?')
    args = parser.parse_args()    

    flightpath1 = parseFlightFile(args.flightfile1)
    flightpath2 = parseFlightFile(args.flightfile2)

    map_name = args.mapname
    
    # ----- flight track -----
    
    mark_static_google_map(mapname=args.mapname, flightpath1=flightpath1, flightpath2=flightpath2)


# -----------------------------------------------------------------------------
# run script

if __name__ == '__main__':
    main()  
    
