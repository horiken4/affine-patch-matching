#include "util.hpp"

void Util::ComputeHomographyFromCameraPose(const Matx33d& int_mat,
                                           const Pose& pose,
                                           Matx33d& homography) {
    auto p = int_mat * pose.mat();

    homography = Matx33d::zeros();
    Mat(p, false).colRange(0, 2).copyTo(Mat(homography, false).colRange(0, 2));
    Mat(p, false).colRange(3, 4).copyTo(Mat(homography, false).colRange(2, 3));
    homography *= (1.0 / homography(2, 2));
}

void Util::SynthesizeCameraImage(const Matx33d& int_mat, const Mat& obj_img,
                                 const Size& cam_img_size, float inter_coeff,
                                 Mat& cam_img, Pose& pose) {
    Vec3d begin_tv = Vec3f(0, 0, 1);
    Vec3d end_tv = Vec3f(0.2, 0.2, 2);

    Vec3d tvec = begin_tv + (end_tv - begin_tv) * inter_coeff;
    Vec3d rvec = normalize(Vec3d(inter_coeff - 1, inter_coeff, 0)) *
                 ((70 * inter_coeff) * M_PI / 180);

    pose.set_rvec(rvec);
    pose.set_tvec(tvec);

    SynthesizeCameraImage(int_mat, obj_img, cam_img_size, pose, cam_img);
}

void Util::SynthesizeCameraImage(const Matx33d& int_mat, const Mat& obj_img,
                                 const Size& cam_img_size, const Pose pose,
                                 Mat& cam_img) {
    Matx33d homo;
    ComputeHomographyFromCameraPose(int_mat, pose, homo);

    Matx33d register_mat(1.0 / obj_img.cols, 0, -0.5, 0, 1.0 / obj_img.rows,
                         -0.5, 0, 0, 1);
    warpPerspective(obj_img, cam_img, homo * register_mat, cam_img_size,
                    INTER_LINEAR, BORDER_CONSTANT, Scalar(80, 80, 80));
}

void Util::ExtractPointsFromKeyPoints(const vector<KeyPoint>& kps,
                                      vector<Point2f>& ps) {
    ps.clear();
    ps.reserve(kps.size());

    for (auto kp : kps) {
        ps.push_back(kp.pt);
    }
}

void Util::FindExtrema(const Mat& surface, Point2f& extrema) {
    Mat A(surface.rows * surface.cols, 5, CV_32FC1);
    Mat b(surface.rows * surface.cols, 1, CV_32FC1);

    for (auto y = 0; y < surface.rows; ++y) {
        for (auto x = 0; x < surface.cols; ++x) {
            auto i = y * surface.rows + x;
            A.at<float>(i, 0) = x * x;
            A.at<float>(i, 1) = y * y;
            A.at<float>(i, 2) = x;
            A.at<float>(i, 3) = y;
            A.at<float>(i, 4) = 1;

            b.at<float>(i, 0) = surface.at<float>(y, x);
        }
    }

    Mat x = (A.t() * A).inv() * A.t() * b;

    auto aa = 1.0f / x.at<float>(0, 0);
    auto bb = 1.0f / x.at<float>(1, 0);
    extrema.x = -aa / 2 * x.at<float>(2, 0);
    extrema.y = -bb / 2 * x.at<float>(3, 0);
}
