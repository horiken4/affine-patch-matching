
#include "referenceframe.hpp"

ReferenceFrame::ReferenceFrame(const vector<Mat>& img_pyramid,
                               const vector<vector<KeyPoint>>& kps_pyramid)
    : Frame(img_pyramid, kps_pyramid) {
    auto conf = Config::Get();

    // Represent all pyramid layer keypoints on 0-level layer
    for (auto level = 0; level < FrameFactory::kPyramidSize; ++level) {
        vector<Point2f> pts;
        Util::ExtractPointsFromKeyPoints(kps_pyramid.at(level), pts);
        auto s = Matx33d(pow(2, level), 0, 0, 0, pow(2, level), 0, 0, 0, 1);
        perspectiveTransform(pts, pts, conf.int_mat().inv() * s);

        obj_pts_.insert(end(obj_pts_), begin(pts), end(pts));
    }

    real_obj_pts_.reserve(obj_pts_.size());
    for (auto pt : obj_pts_) {
        real_obj_pts_.push_back(Point3f(pt.x, pt.y, 0));
    }
}

ReferenceFrame::~ReferenceFrame() {}

const vector<Point2f>& ReferenceFrame::obj_pts() const { return obj_pts_; }

const vector<Point3f>& ReferenceFrame::real_obj_pts() const {
    return real_obj_pts_;
}

