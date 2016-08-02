
# brief : bilateral filtering images with different parameters, match the generated images with the corresponding satellite map and write the details into a json file

# -----------------------------------------------------------------------------
# import modules

import cv2
import numpy as np
import argparse
import os
import glob
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def flann_match(query_image, train_image, threshold):
    
    query = cv2.imread(query_image, 0)
    train = cv2.imread(train_image, 0)  ## best grid in database
    surf = cv2.SURF(threshold)
    kp1, desc1 = surf.detectAndCompute(query, None)
    kp2, desc2 = surf.detectAndCompute(train, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.asarray(desc1,np.float32),np.asarray(desc2,np.float32),k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    threshold_list = [i for i in np.arange(0.5, 0.8, .05)]
    position = 4
    # print(threshold_list[position])
    for m,n in matches:
        if m.distance < threshold_list[position]*n.distance:
            good.append(m)

    return kp1, kp2, good


def bilateral_processing(image, par):
    img = cv2.imread(image)
    blur = cv2.bilateralFilter(img,par,75,75)
    image_name = "bl_" + str(par) + '_' + os.path.basename(image)
    cv2.imwrite(image_name, blur)
    #print(image_name)


def main():
    
   # ----- parse arguments -----
   
   parser = argparse.ArgumentParser(description='Parsing')
   parser.add_argument('image_name', help='require the query image')
   parser.add_argument('query_dir', help='requires a query image directory')
   parser.add_argument('train', help='requires a train image')
   parser.add_argument('threshold', default=3200, nargs='?')  ## hessian threshold
   
   args = parser.parse_args()
   
   # ----- bilateral filtering -----
   
   image = args.image_name
   para_list = [5, 10, 15, 20, 25]  ## the diameter of each pixel neighborhood
   for para in para_list:
       bilateral_processing(image, para)

   # ----- matching all -----

   query_image_dir = args.query_dir
   train_image = args.train
   threshold = int(args.threshold)
   query_images_list = glob.glob(os.path.join(query_image_dir, '*.png'))
   rate_list = list()
   para_list = list()
   summary = dict()
   for query_image in query_images_list:
        
        kp1, kp2, matches = flann_match(query_image, train_image, threshold)
        matching_rate = len(matches) / float(len(kp1))
        rate_list.append(matching_rate)
        #print(os.path.basename(query_image))
        d = os.path.basename(query_image)
        name = os.path.splitext(d)[0]
        st = name.split('_')[1]
        para_list.append(st)
        summary[os.path.basename(query_image)] = matching_rate

   directory = os.getcwd()
   file_o = "{0}.json".format('matching_performance')
   dst_path = os.path.join(directory, file_o)
   with open(dst_path, "w") as fo:
        fo.write(json.dumps(summary))


if __name__ == '__main__':
   main()

