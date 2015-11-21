#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include <opencv2/opencv.hpp>

#include "frame.hpp"

using namespace std;
using namespace cv;

class Util {
   public:
    static void ComputeHomographyFromCameraPose(const Matx33d& int_mat,
                                                const Pose& pose,
                                                Matx33d& homography);

    static void SynthesizeCameraImage(const Matx33d& int_mat,
                                      const Mat& obj_img,
                                      const Size& cam_img_size,
                                      float inter_coeff, Mat& cam_img,
                                      Pose& pose);

    static void SynthesizeCameraImage(const Matx33d& int_mat,
                                      const Mat& obj_img,
                                      const Size& cam_img_size, const Pose pose,
                                      Mat& cam_img);

    static void ExtractPointsFromKeyPoints(const vector<KeyPoint>& kps,
                                           vector<Point2f>& ps);

    static void FindExtrema(const Mat& surface, Point2f& extrema);
};

#endif
