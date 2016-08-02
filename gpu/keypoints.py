# author: Matthew Triche
# brief : This python module is designed to assist with loading JSON data containing keypoint information.

# -----------------------------------------------------------------------------
# import modules

import json
import os

# -----------------------------------------------------------------------------
# class KeyPoint
#
# This class is for constructing keypoint from JSON data.

class KeyPoint:
	
	# -------------------------------------------------------------------------
	# __init__
	#
	# Constructor
	#
	# Parameters:
	# dstruct - The data structure loaded from JSON which contains data for a single keypoint.
	
	def __init__(self,dstruct):
		
		self.pt       = dstruct[0]
		self.size     = dstruct[1]
		self.angle    = dstruct[2]
		self.response = dstruct[3]
		self.octave   = dstruct[4]
		self.class_id = dstruct[5]
		self.desc     = dstruct[6] 

# -----------------------------------------------------------------------------
# function LoadJSON
#
# Load KeyPoints from a JSON file.
#
# Parameters:
# filename - Filename where the JSON data can be located.
#
# Return Value:
# An empty list is returned if the JSON data could not be loaded for any reason.
# A list populated with instances of class KeyPoint is returned otherwise.

def LoadJSON(filename):
	
	if os.path.exists(filename):
		fh = open(filename,"r")
	else:
		print("Error: LoadJSON: Unable to open file " + filename)
		return []
	
	try:
		json_data = json.loads(fh.read())
	except ValueError as e:
		print("Error: LoadJSON: Unable to load json data: " + str(e))
		return []
	
	return [ KeyPoint(dstruct) for dstruct in json_data.values() ]

	
