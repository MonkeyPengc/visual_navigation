
# brief : this script contains a class used to manage c library and handle calls from python

# in .bashrc file, export VISUAL_NAV_PROJECT_GPU={the folder path of gpu code}

# -----------------------------------------------------------------------------
# import modules

from ctypes import *
from c_cv2 import *
import os


lib_filename = os.path.join(os.environ['VISUAL_NAV_PROJECT_GPU'], "_libng.so")
lib = cdll.LoadLibrary(lib_filename)  # create handle to libng.so


# ----- a class that calls external functions -----

class LibNg:
    
    def __init__(self):
        self.lib = lib
        
        # ----- declare argument types for external functions within _libng.so -----
        
        self.lib.ConfigureSURF.argtypes = [c_int]
        self.lib.LoadFrameImage.argtypes = [c_char_p]
        self.lib.LoadGridKeypoints.argtypes = [POINTER(c_KeyPoint), c_int]
        self.lib.LoadGridDescriptors.argtypes = [POINTER(c_float), c_int, c_int]
        self.lib.Process.argtypes = [POINTER(c_Matches), c_double]
        self.lib.GetMatchPositions.argtypes = [POINTER(c_Point2f)]
    
        # ----- declare return types when external function within _libng.so are called -----
        self.lib.LoadFrameImage.restype = c_bool


    # ----- class destructor -----
    
    def __del__(self):
        self.lib.Release()


    # ----- define calls to each external function within _libng.so -----

    def ConfigureSURF(self, minHess):
        self.lib.ConfigureSURF(minHess)

    def LoadFrameImage(self, filename):
        return self.lib.LoadFrameImage(filename)

    def LoadGridKeypoints(self, c_kp):
        ptr = cast(pointer(c_kp), POINTER(c_KeyPoint)) # aquire a pointer to the array of keypoints
        return self.lib.LoadGridKeypoints(ptr, len(c_kp))
        
    def LoadGridDescriptors(self, c_desc, dim):
        ptr = cast(pointer(c_desc), POINTER(c_float)) # aquire a pointer to the array of descriptors
        return self.lib.LoadGridDescriptors(ptr, dim, len(c_desc) / dim)

    def Process(self, c_matches, ratio):
        ptr = cast(pointer(c_matches), POINTER(c_Matches)) # aquire a pointer to the structure where results will be stored
        self.lib.Process(ptr, ratio)

    def GetMatchPositions(self, c_positions):
        ptr = cast(pointer(c_positions), POINTER(c_Point2f))
        self.lib.GetMatchPositions(ptr)





