#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "util.hpp"
#include "framefactory.hpp"
#include "frame.hpp"
#include "referenceframe.hpp"
#include "patchfinder.hpp"

using namespace cv;
using namespace std;

Mat object_image;

void OnFrameChange(int pos, void* userdata) {
    Mat* cam_img = static_cast<Mat*>(userdata);
    Pose pose;
    auto conf = Config::Get();
    Util::SynthesizeCameraImage(conf.int_mat(), object_image,
                                conf.cam_img_size(), pos / 100.0, *cam_img,
                                pose);
    imshow("camera", *cam_img);
}

int main() {
    Mat obj_img = imread("./lena.jpg");
    object_image = obj_img;
    Mat cam_img;
    namedWindow("camera");
    int scroll_val = 0;
    createTrackbar("frame", "camera", &scroll_val, 100, OnFrameChange,
                   &cam_img);
    waitKey(0);

    auto conf = Config::Get();

    FrameFactory frame_factory;
    PatchFinder patch_finder(conf.int_mat());

    // Make reference frame
    Mat ref_img;
    Mat ref_frame_img;
    shared_ptr<Frame> ref_frame;
    Pose ref_pose;
    Util::SynthesizeCameraImage(conf.int_mat(), obj_img, conf.cam_img_size(), 0,
                                ref_img, ref_pose);
    cvtColor(ref_img, ref_img, COLOR_BGR2GRAY);
    frame_factory.Make(ref_img, ref_frame, Frame::kKindReference);
    ref_frame->Visualize(ref_frame_img);

    /*
    namedWindow("ref img");
    imshow("ref img", ref_img);
    namedWindow("ref frame");
    imshow("ref frame", ref_frame_img);
    */

    // Make previous frames for predict next frame pose
    vector<float> prev_inter_coeffs = {0.5, 0.6};
    vector<Mat> prev_imgs;
    vector<Mat> prev_frame_imgs;
    vector<shared_ptr<Frame>> prev_frames;
    vector<Pose> prev_poses;

    for (auto inter_coeff : prev_inter_coeffs) {
        Mat img;
        Mat frame_img;
        shared_ptr<Frame> frame;
        Pose pose;

        Util::SynthesizeCameraImage(conf.int_mat(), obj_img,
                                    conf.cam_img_size(), inter_coeff, img,
                                    pose);
        cvtColor(img, img, COLOR_BGR2GRAY);
        frame_factory.Make(img, frame);
        frame->Visualize(frame_img);

        prev_imgs.push_back(img);
        prev_frame_imgs.push_back(frame_img);
        prev_frames.push_back(frame);
        prev_poses.push_back(pose);

        /*
        namedWindow("prev img");
        imshow("prev img", img);
        namedWindow("prev frame");
        imshow("prev frame", frame_img);
        waitKey(0);
        */
    }

    // Make target frame you want to predict pose
    float target_inter_coeff = 0.74;
    Mat target_img;
    Mat target_frame_img;
    shared_ptr<Frame> target_frame;
    Pose target_pose;
    Util::SynthesizeCameraImage(conf.int_mat(), obj_img, conf.cam_img_size(),
                                target_inter_coeff, target_img, target_pose);
    cvtColor(target_img, target_img, COLOR_BGR2GRAY);
    frame_factory.Make(target_img, target_frame);
    target_frame->Visualize(target_frame_img);

    // Make affine patch and put patches on target frame
    vector<Point2f> search_origins;
    vector<int> search_pyramid_levels;
    vector<Mat> patches;
    patch_finder.MakePatch(dynamic_pointer_cast<ReferenceFrame>(ref_frame),
                           target_pose, search_pyramid_levels, search_origins,
                           patches);

    Mat target_patch_img = target_img.clone();
    for (auto i = 0; i < patches.size(); i++) {
        auto c = search_origins.at(i);
        auto level = search_pyramid_levels.at(i);
        auto patch = patches.at(i);

        // Adjust scale and center according to level
        auto level_c = c * pow(2, level);
        Mat level_patch;
        auto size = PatchFinder::kPatchSize * pow(2, level);
        resize(patch, level_patch, Size(size, size));

        Rect patch_roi(level_c.x - level_patch.cols / 2,
                       level_c.y - level_patch.rows / 2, level_patch.cols,
                       level_patch.rows);
        if (patch_roi.x < 0 || patch_roi.y < 0 ||
            patch_roi.x + patch_roi.width >= target_patch_img.cols ||
            patch_roi.y + patch_roi.height >= target_patch_img.rows) {
            continue;
        }
        level_patch.copyTo(target_patch_img(patch_roi));
    }

    // Predict target frame extrinsic matrix with linear motion model
    // FIXME: Is this correct? Is usage of rodrigues correct?
    auto target_prior_rvec = prev_poses.at(1).rvec() + prev_poses.at(1).rvec() -
                             prev_poses.at(0).rvec();
    auto target_prior_tvec = prev_poses.at(1).tvec() + prev_poses.at(1).tvec() -
                             prev_poses.at(0).tvec();
    Pose target_prior_pose(target_prior_rvec, target_prior_tvec);

    cout << "GT target pose:" << endl;
    cout << target_pose.mat() << endl;

    cout << "-----" << endl;
    cout << "target prior pose:" << endl;
    cout << target_prior_pose.mat() << endl;
    cout << "diff(L1-norm)="
         << norm(target_pose.mat() - target_prior_pose.mat(), NORM_L1) << endl;

    // Estimate target frame extrinsic matrix
    Pose estimated_target_pose;
    Mat estimated_target_img;
    auto succeeded = patch_finder.EstimatePose(
        ref_frame, target_frame, target_prior_pose, estimated_target_pose);
    cout << "-----" << endl;
    cout << "Estimated target pose:" << endl;
    if (succeeded) {
        cout << estimated_target_pose.mat() << endl;
        cout << "diff(L1-norm)="
             << norm(target_pose.mat() - estimated_target_pose.mat(), NORM_L1)
             << endl;
    } else {
        cout << "Failed to estimate" << endl;
    }
    Util::SynthesizeCameraImage(conf.int_mat(), object_image,
                                conf.cam_img_size(), estimated_target_pose,
                                estimated_target_img);

    // Estimate target frame extrinsic matrix by corase-to-fine approach
    Pose ctf_estimated_target_pose;
    Mat ctf_estimated_target_img;
    succeeded = patch_finder.EstimatePoseCoarseToFine(
        ref_frame, target_frame, target_prior_pose, ctf_estimated_target_pose);
    cout << "-----" << endl;
    cout << "Estimated target pose (coarse-to-fine):" << endl;
    if (succeeded) {
        cout << ctf_estimated_target_pose.mat() << endl;
        cout << "diff(L1-norm)="
             << norm(target_pose.mat() - ctf_estimated_target_pose.mat(),
                     NORM_L1)
             << endl;
    } else {
        cout << "Failed to estimate" << endl;
    }
    Util::SynthesizeCameraImage(conf.int_mat(), object_image,
                                conf.cam_img_size(), ctf_estimated_target_pose,
                                ctf_estimated_target_img);

    // Visualize estimated target scena
    namedWindow("target patch img");
    imshow("target patch img", target_patch_img);

    namedWindow("GT target img");
    imshow("GT target img", target_img);

    namedWindow("estimated target img");
    imshow("estimated target img", estimated_target_img);

    namedWindow("estimated target img (coarse-to-fine)");
    imshow("estimated target img (coarse-to-fine)", ctf_estimated_target_img);

    waitKey(0);
}
