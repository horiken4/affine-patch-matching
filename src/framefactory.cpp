#include "framefactory.hpp"

const int FrameFactory::kPyramidSize = 4;

FrameFactory::FrameFactory()
    : feature_detector_(FastFeatureDetector::create()) {}

FrameFactory::~FrameFactory() {}

void FrameFactory::Make(const Mat& cam_img, shared_ptr<Frame>& frame, Frame::Kind kind) {
    // Make image pyramid
    vector<Mat> img_pyramid(kPyramidSize, Mat());
    Mat img = cam_img.clone();
    for (int level = 0; level < kPyramidSize; ++level) {
        img_pyramid.at(level) = img.clone();
        pyrDown(img, img,
                Size(img.cols / 2, img.rows / 2));
    }

    // Extract FAST
    vector<vector<KeyPoint>> kps_pyramid(kPyramidSize, vector<KeyPoint>());
    for (int level = 0; level < kPyramidSize; ++level) {
        feature_detector_->detect(img_pyramid.at(level), kps_pyramid.at(level));
        KeyPointsFilter::retainBest(kps_pyramid.at(level), 100);
    }

    if (kind == Frame::kKindNormal) {
        frame = shared_ptr<Frame>(new Frame(img_pyramid, kps_pyramid));
    } else if (kind == Frame::kKindReference) {
        frame = shared_ptr<Frame>(new ReferenceFrame(img_pyramid, kps_pyramid));
    }
}
