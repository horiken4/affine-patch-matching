#ifndef FRAME_HPP
#define FRAME_HPP

#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

#include "pose.hpp"

using namespace std;
using namespace cv;

class Frame {
   public:
    enum Kind {
        kKindNormal,
        kKindReference,
    };

   public:
    Frame(const vector<Mat>& img_pyramid,
          const vector<vector<KeyPoint>>& kps_pyramid);
    virtual ~Frame();

    void Visualize(Mat& img);

    const vector<Mat>& img_pyramid() const;
    const vector<vector<KeyPoint>>& kps_pyramid() const;

   protected:
    vector<Mat> img_pyramid_;
    vector<vector<KeyPoint>> kps_pyramid_;
};

#endif
