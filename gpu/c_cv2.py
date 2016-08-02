
# brief : Implements OpenCV structures using ctypes. 


# -----------------------------------------------------------------------------
# Import Modules

from ctypes import *


# -----------------------------------------------------------------------------
# define c_Point2f structure

class c_Point2f(Structure):
	_fields_ = [ ("x", c_float), ("y", c_float) ]


# -----------------------------------------------------------------------------
# define c_KeyPoint structure

class c_KeyPoint(Structure):
	_fields_ = [ ("_pt",       c_Point2f), 
	             ("_size",     c_float),
	             ("_angle",    c_float),
	             ("_response", c_float),
	             ("_octave",   c_int),
	             ("_class_id", c_int) ]


# -----------------------------------------------------------------------------
# define c_Matches structure that is used to obtain results from _libng.so

class c_Matches(Structure):
    _fields_ = [("ftime", c_double),
                ("mtime", c_double),
                ("m_num", c_int)
                ]

