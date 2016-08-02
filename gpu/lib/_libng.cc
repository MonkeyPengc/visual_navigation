
// brief: a library that contains external "C" functions using gpu opencv library.

/* ------------------------------------------------------------------------- *
 * Include Header Files                                                      *
 * ------------------------------------------------------------------------- */

#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"


/* ------------------------------------------------------------------------- *
 * Define Namespaces                                                         *
 * ------------------------------------------------------------------------- */

using namespace std;
using namespace cv;
using namespace cv::gpu;


/* ------------------------------------------------------------------------- *
 * Define Types                                                              *
 * ------------------------------------------------------------------------- */

typedef struct C_POINT2F_
{
    float x,y;
} c_Point2f;


typedef struct C_KEYPOINT_
{
    c_Point2f _pt;
    float _size;
    float _angle;
    float _response;
    int   _octave;
    int   _class_id;
} c_KeyPoint;


typedef struct MATCHES_
{
    double ftime;      // the elapsed time of the frame feature phase. (seconds)
    double mtime;      // the elapsed time of the matching phase. (seconds)
    int m_num;         // stores the number of matches made.
} matches_t;



/* ------------------------------------------------------------------------- *
 * Declare External Functions                                                *
 * ------------------------------------------------------------------------- */

extern "C" void ConfigureSURF(int);
extern "C" void LoadGridKeypoints(c_KeyPoint*, int);
extern "C" void LoadGridDescriptors(float*, int, int);
extern "C" bool LoadFrameImage(char *filename);
extern "C" void Process(matches_t*, double);
extern "C" void GetMatchPositions(c_Point2f*);
extern "C" void Release();

/* ------------------------------------------------------------------------- *
 * Define Internal Variables                                                 *
 * ------------------------------------------------------------------------- */

static Mat FrameDesc, GridDesc;
static Mat FrameImg;
static vector<KeyPoint> FrameKeypoints, GridKeypoints;
static vector<DMatch> matches;
static SURF_GPU *surf = NULL;
static GpuMat FrameDescGPU, GridDescGPU;
static GpuMat FrameImgGPU;
static GpuMat FrameKeypointsGPU, GridKeypointsGPU;


/* ------------------------------------------------------------------------- *
 * Define External Functions                                                 *
 * ------------------------------------------------------------------------- */

extern "C" void ConfigureSURF(int minHess)
{
    #define SURF_N_OCT        4
    #define SURF_N_OCT_LAYERS 2
    #define SURF_EXTENDED     false
    #define KP_RATIO          0.01
    
    if(surf)
    {
        surf->releaseMemory();
        delete surf;
    }

    surf = new SURF_GPU(minHess,
                        SURF_N_OCT,
                        SURF_N_OCT_LAYERS,
                        SURF_EXTENDED,
                        KP_RATIO);
}


extern "C" bool LoadFrameImage(char *filename)
{
    FrameImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    
    if(!FrameImg.empty())
    FrameImgGPU.upload(FrameImg);
    
    return !FrameImg.empty();
}


extern "C" void LoadGridKeypoints(c_KeyPoint *buff, int num)
{
    GridKeypoints.resize(num);
    
    for(int i = 0; i < num; i++)
    {
        GridKeypoints[i] = KeyPoint(Point2f(buff[i]._pt.x, buff[i]._pt.y),
                                     buff[i]._size,
                                     buff[i]._angle,
                                     buff[i]._response,
                                     buff[i]._octave,
                                     buff[i]._class_id);
    }
    surf->uploadKeypoints(GridKeypoints, GridKeypointsGPU);
}


extern "C" void LoadGridDescriptors(float *buff, int dim, int num)
{
    GridDesc.create(num, dim, CV_32F);
        
    /* there is a much faster way to do this */
    for(int c = 0; c < num; c++)
    {
        for(int r = 0; r < dim; r++)
        {
            GridDesc.at<float>(c,r) = *(buff++);
        }
    }
    GridDescGPU.upload(GridDesc);
}


extern "C" void Process(matches_t *dst, double ratio)
{

    int tstart, tend;  // time counter
    
    tstart = clock();
    
    //detect keypoints and descriptors of frame
    (*surf)(FrameImgGPU, GpuMat(), FrameKeypointsGPU, FrameDescGPU);
    
    tend = clock();
    dst->ftime = (double)(tend - tstart)/(double)CLOCKS_PER_SEC; // stores detection time
    
    //each init_matches[i] is k or less matches for the same query descriptor
    vector< vector<DMatch> > init_matches;
    
    BFMatcher_GPU matcher(NORM_L2);
    
    tstart = clock();
    
    //generate matches
    matcher.knnMatch(FrameDescGPU, GridDescGPU, init_matches, 2);
    matches.clear();
    
    //push good matches to vector
    for (int i = 0; i < init_matches.size(); i++)
    {
        if(init_matches[i][0].distance < (ratio*init_matches[i][1].distance))
        {
            matches.push_back(init_matches[i][0]);
        }
    }
    
    tend = clock();
    dst->mtime = (double)(tend - tstart)/(double)CLOCKS_PER_SEC;  // stores matching time
    
    dst->m_num = matches.size(); // stores number of matches found
}


extern "C" void GetMatchPositions(c_Point2f *dst)
{

    //dst = new c_Point2f[matches.size()];
    for(int i = 0; i < matches.size(); i++)
    {
        dst[i].x = GridKeypoints[matches[i].trainIdx].pt.x;  // stores the positions of matches
        dst[i].y = GridKeypoints[matches[i].trainIdx].pt.y;
    }

}


extern "C" void Release()
{
    GridDesc.release();
    FrameDesc.release();
    FrameImg.release();
    
    GridKeypoints.clear();
    FrameKeypoints.clear();
    matches.clear();
    
    GridDescGPU.release();
    FrameDescGPU.release();
    FrameImgGPU.release();
    FrameKeypointsGPU.release();
    GridKeypointsGPU.release();
    
    surf->releaseMemory();
    delete surf;


}
    



