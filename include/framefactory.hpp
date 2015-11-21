#ifndef FRAME_FACTORY_HPP
#define FRAME_FACTORY_HPP

#include <vector>
#include <opencv2/opencv.hpp>

#include "frame.hpp"
#include "referenceframe.hpp"

using namespace std;
using namespace cv;

class FrameFactory {
   public:
    static const int kPyramidSize;

   public:
    FrameFactory();
    ~FrameFactory();

    void Make(const Mat& cam_img, shared_ptr<Frame>& frame, Frame::Kind kind = Frame::kKindNormal);

   private:
    Ptr<FastFeatureDetector> feature_detector_;
};

#endif
