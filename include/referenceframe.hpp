
#ifndef REFERENCE_FRAME_HPP
#define REFERENCE_FRAME_HPP

#include <vector>
#include <opencv2/opencv.hpp>

#include "frame.hpp"
#include "config.hpp"
#include "util.hpp"
#include "framefactory.hpp"
#include "pose.hpp"

using namespace std;
using namespace cv;

class ReferenceFrame : public Frame {
   public:
    ReferenceFrame(const vector<Mat>& img_pyramid,
                   const vector<vector<KeyPoint>>& kps_pyramid);
    ~ReferenceFrame();

    const vector<Point2f>& obj_pts() const;
    const vector<Point3f>& real_obj_pts() const;

   private:
    vector<Point2f> obj_pts_;
    vector<Point3f> real_obj_pts_;
};

#endif
