#include "frame.hpp"
#include "referenceframe.hpp"

Frame::Frame(const vector<Mat>& img_pyramid,
             const vector<vector<KeyPoint>>& kps_pyramid)
    : img_pyramid_(img_pyramid), kps_pyramid_(kps_pyramid) {}

Frame::~Frame() {}

void Frame::Visualize(Mat& img) {
    Size canvas_size(0, 0);
    for (auto img : img_pyramid_) {
        canvas_size.width += img.cols;
        canvas_size.height += img.rows;
    }
    Mat canvas(canvas_size, CV_8UC3);

    int canvas_offset_y = 0;
    for (auto level = 0; level < img_pyramid_.size(); ++level) {
        Mat kp_img = img_pyramid_.at(level).clone();
        cvtColor(kp_img, kp_img, COLOR_GRAY2BGR);
        drawKeypoints(kp_img, kps_pyramid_.at(level), kp_img, Scalar::all(-1),
                      DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        kp_img.copyTo(
            canvas(Rect(0, canvas_offset_y, kp_img.cols, kp_img.rows)));
        canvas_offset_y += kp_img.rows;
    }

    img = canvas;
}

const vector<Mat>& Frame::img_pyramid() const { return img_pyramid_; }
const vector<vector<KeyPoint>>& Frame::kps_pyramid() const {
    return kps_pyramid_;
}
