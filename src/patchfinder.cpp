#include "patchfinder.hpp"

const int PatchFinder::kPatchSize = 8;
const int PatchFinder::kCoarsePatchSearchRadius = 6;
const int PatchFinder::kFinePatchSearchRadius = 3;
const int PatchFinder::kNumCoarseMatches = 25;
const int PatchFinder::kNumFineMatches = 100;

PatchFinder::PatchFinder(const Matx33d& int_mat) : int_mat_(int_mat) {}

PatchFinder::~PatchFinder() {}

bool PatchFinder::EstimatePose(const shared_ptr<Frame>& ref_frame,
                               const shared_ptr<Frame>& target_frame,
                               const Pose& prior_pose, Pose& pose) {
    auto reference_frame = dynamic_pointer_cast<ReferenceFrame>(ref_frame);

    // Make affine warped patch
    vector<Point2f> patch_pts;
    vector<int> patch_pyramid_levels;
    vector<Mat> patches;
    MakePatch(reference_frame, prior_pose, patch_pyramid_levels, patch_pts,
              patches);

    vector<Point3f> found_real_obj_pts;
    vector<Point2f> found_img_pts;
    found_real_obj_pts.reserve(reference_frame->obj_pts().size());
    found_img_pts.reserve(reference_frame->obj_pts().size());

    // Find location of all patches
    for (auto i = 0; i < patches.size(); ++i) {
        auto patch_pyramid_level = patch_pyramid_levels.at(i);
        auto patch_pt = patch_pts.at(i);
        Mat find_img = target_frame->img_pyramid().at(patch_pyramid_level);
        Mat patch = patches.at(i);

        // Define search ROI
        const int kPatchSearchRadius = 3;
        Point2f find_roi_tl(
            patch_pt.x - kPatchSearchRadius - kPatchSize / 2.0f,
            patch_pt.y - kPatchSearchRadius - kPatchSize / 2.0f);
        Point2f find_roi_br(
            patch_pt.x + kPatchSearchRadius + kPatchSize / 2.0f,
            patch_pt.y + kPatchSearchRadius + kPatchSize / 2.0f);

        if (find_roi_tl.x < 0 || find_roi_tl.y < 0 ||
            find_roi_br.x >= find_img.cols || find_roi_br.y >= find_img.rows) {
            // If search ROI is outside of this frame, ignore this point
            continue;
        }

        Rect find_roi(find_roi_tl, find_roi_br);

        // Find patch location on 0-level pyramid layer
        Mat zncc;
        Point found_roi_pt;
        matchTemplate(find_img(find_roi), patch, zncc, TM_CCOEFF_NORMED);
        minMaxLoc(zncc, NULL, NULL, NULL, &found_roi_pt);

        Point2f found_img_pt(
            (found_roi_pt.x - kPatchSearchRadius) + patch_pt.x,
            (found_roi_pt.y - kPatchSearchRadius) + patch_pt.y);
        found_img_pt *= pow(2, patch_pyramid_level);

        found_img_pts.push_back(found_img_pt);
        found_real_obj_pts.push_back(reference_frame->real_obj_pts().at(i));
    }

    if (patch_pts.size() < 10) {
        return false;
    }

    // Compute extrinsic matrix of this frame
    Vec3d rvec = prior_pose.rvec();
    Vec3d tvec = prior_pose.tvec();
    solvePnP(found_real_obj_pts, found_img_pts, int_mat_, noArray(), rvec, tvec,
             true, SOLVEPNP_ITERATIVE);

    pose.set_rvec(rvec);
    pose.set_tvec(tvec);

    return true;
}

bool PatchFinder::EstimatePoseCoarseToFine(
    const shared_ptr<Frame>& ref_frame, const shared_ptr<Frame>& target_frame,
    const Pose& prior_pose, Pose& pose) {
    // 1. At coarse level, coarse patch location is found. The location is
    // subpixel accuracy
    // 2. Coarse pose is estimated by coarse patch locations
    // 3. Update fine level search origins by coarse pose
    // 4. At fine level, fine patch location is found
    // 5. Final pose is estimated by coarse and fine patch locations

    auto reference_frame = dynamic_pointer_cast<ReferenceFrame>(ref_frame);

    // Make affine warped patch

    vector<Point2f> patch_pts;
    vector<int> patch_pyramid_levels;
    vector<Mat> patches;
    MakePatch(reference_frame, prior_pose, patch_pyramid_levels, patch_pts,
              patches);

    // Decide patches used as matching randomly
    auto obj_pts = reference_frame->obj_pts();
    auto num_obj_pts = obj_pts.size();
    vector<int> coarse_indices;
    vector<int> fine_indices;
    coarse_indices.reserve(num_obj_pts / 2);
    fine_indices.reserve(num_obj_pts / 2);
    for (auto i = 0; i < patch_pyramid_levels.size(); ++i) {
        auto level = patch_pyramid_levels.at(i);
        if (level == 0) {
            // Fine (0-level) layer
            fine_indices.push_back(i);
        } else if (level == 1) {
            // Coarse (1-level) layer
            coarse_indices.push_back(i);
        }
    }
    auto seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(begin(coarse_indices), end(coarse_indices),
            default_random_engine(seed));
    shuffle(begin(fine_indices), end(fine_indices),
            default_random_engine(seed));

    if (coarse_indices.size() > kNumCoarseMatches) {
        coarse_indices.erase(begin(coarse_indices) + kNumCoarseMatches,
                             end(coarse_indices));
    }
    if (fine_indices.size() > kNumFineMatches) {
        fine_indices.erase(begin(fine_indices) + kNumFineMatches,
                           end(fine_indices));
    }

    // Find patch at coarse level
    vector<Point3f> found_real_obj_pts;
    vector<Point2f> found_img_pts;
    auto succeeded =
        FindPatch(reference_frame, target_frame, patch_pyramid_levels,
                  patch_pts, patches, coarse_indices, kCoarsePatchSearchRadius,
                  found_real_obj_pts, found_img_pts);
    if (!succeeded) {
        return false;
    }

    // Compute extrinsic matrix from coarse match
    Vec3d rvec = prior_pose.rvec();
    Vec3d tvec = prior_pose.tvec();
    solvePnP(found_real_obj_pts, found_img_pts, int_mat_, noArray(), rvec, tvec,
             true, SOLVEPNP_ITERATIVE);

    // Update search origins abount only fine indices by coarse rvec, tvec
    auto real_obj_pts = reference_frame->real_obj_pts();
    vector<Point3f> fine_real_obj_pts;
    vector<Point2f> fine_patch_pts;
    fine_real_obj_pts.reserve(fine_indices.size());
    fine_patch_pts.reserve(fine_indices.size());
    for (auto idx : fine_indices) {
        fine_real_obj_pts.push_back(real_obj_pts.at(idx));
    }
    projectPoints(fine_real_obj_pts, rvec, tvec, int_mat_, noArray(),
                  fine_patch_pts);
    for (auto i = 0; i < fine_indices.size(); ++i) {
        auto idx = fine_indices.at(i);
        patch_pts.at(idx) = fine_patch_pts.at(i);
    }

    // Find patch at fine level
    succeeded =
        FindPatch(reference_frame, target_frame, patch_pyramid_levels,
                  patch_pts, patches, fine_indices, kFinePatchSearchRadius,
                  found_real_obj_pts, found_img_pts);
    if (!succeeded) {
        return false;
    }

    // Compute extrinsic matrix from fine match
    solvePnP(found_real_obj_pts, found_img_pts, int_mat_, noArray(), rvec, tvec,
             true, SOLVEPNP_ITERATIVE);
    pose.set_rvec(rvec);
    pose.set_tvec(tvec);

    return true;
}

void PatchFinder::MakePatch(const shared_ptr<ReferenceFrame>& ref_frame,
                            const Pose& prior_pose,
                            vector<int>& patch_pyramid_levels,
                            vector<Point2f>& patch_pts, vector<Mat>& patches) {
    Matx33d homo;
    Util::ComputeHomographyFromCameraPose(int_mat_, prior_pose, homo);

    for (auto level = 0; level < FrameFactory::kPyramidSize; ++level) {
        auto kps = ref_frame->kps_pyramid().at(level);
        vector<Point2f> pts;
        Util::ExtractPointsFromKeyPoints(kps, pts);

        // Below points is represented as 0-level pyramid layer
        vector<Point2f> frame_pts;
        vector<Point2f> frame_xunit_pts;
        vector<Point2f> frame_yunit_pts;

        // Project points on pyramid level to frame
        Matx33d s = Matx33d(pow(2, level), 0, 0, 0, pow(2, level), 0, 0, 0, 1);
        perspectiveTransform(pts, frame_pts, homo * int_mat_.inv() * s);

        // Project unix x on pyramid level to frame
        auto tx = Matx33d(1, 0, 1, 0, 1, 0, 0, 0, 1);
        perspectiveTransform(pts, frame_xunit_pts,
                             homo * int_mat_.inv() * s * tx);

        // Project unix y on pyramid level to frame
        auto ty = Matx33d(1, 0, 0, 0, 1, 1, 0, 0, 1);
        perspectiveTransform(pts, frame_yunit_pts,
                             homo * int_mat_.inv() * s * ty);

        // TODO: Optimize speed
        for (auto i = 0; i < pts.size(); ++i) {
            auto src_o = pts.at(i);
            vector<Point2f> src = {
                // Reprensetation as each level layer
                src_o, Point2f(src_o.x + 1, src_o.y),
                Point2f(src_o.x, src_o.y + 1),
            };
            vector<Point2f> dst = {
                // Representation as 0-level layer
                frame_pts.at(i), frame_xunit_pts.at(i), frame_yunit_pts.at(i),
            };

            Matx23d affine = getAffineTransform(src, dst);

            // Estimate destination pyramid level
            auto dst_unit_area =
                determinant(Mat(affine, false)(Rect(0, 0, 2, 2)));
            int most_likely_level = -1;
            float most_likely_diff = 10000;
            for (auto level = 0; level < FrameFactory::kPyramidSize; ++level) {
                auto ratio = dst_unit_area / pow(4, level);
                auto diff = abs(1 - ratio);
                if (diff < most_likely_diff) {
                    most_likely_diff = diff;
                    most_likely_level = level;
                }
            }

            // TODO: You should remove patch determinant is small (< 0.8)

            // Make patch
            auto tlans_patch_origin =
                Matx33d(1, 0, -src.at(0).x, 0, 1, -src.at(0).y, 0, 0, 1);
            auto scale = pow(2, -most_likely_level);
            auto patch_affine = Matx33d::eye();
            Mat(affine * scale, false)(Rect(0, 0, 2, 2))
                .copyTo(Mat(patch_affine, false)(Rect(0, 0, 2, 2)));
            auto tlans_patch_center = Matx33d(1, 0, kPatchSize / 2.0f, 0, 1,
                                              kPatchSize / 2.0f, 0, 0, 1);

            patch_affine =
                tlans_patch_center * patch_affine * tlans_patch_origin;

            Mat patch;
            warpAffine(ref_frame->img_pyramid().at(level), patch,
                       Mat(patch_affine, false)(Rect(0, 0, 3, 2)),
                       Size(kPatchSize, kPatchSize));
            patch_pts.push_back(
                dst.at(0) /
                pow(2, most_likely_level));  // Represent pyramid level layer
            patch_pyramid_levels.push_back(most_likely_level);
            patches.push_back(patch);
        }
    }
}

bool PatchFinder::FindPatch(const shared_ptr<ReferenceFrame> ref_frame,
                            const shared_ptr<Frame>& target_frame,
                            const vector<int>& patch_pyramid_levels,
                            const vector<Point2f>& patch_pts,
                            const vector<Mat>& patches,
                            const vector<int>& used_patch_indices,
                            int find_radius,
                            vector<Point3f>& found_real_obj_pts,
                            vector<Point2f>& found_img_pts) {
    auto patch_pyramid_level =
        patch_pyramid_levels.at(used_patch_indices.at(0));

    for (auto i : used_patch_indices) {
        if (patch_pyramid_level != patch_pyramid_levels.at(i)) {
            throw runtime_error("patch_pyramid_level must be same\n");
        }
        auto patch_pt = patch_pts.at(i);
        Mat find_img = target_frame->img_pyramid().at(patch_pyramid_level);
        Mat patch = patches.at(i);

        // Define find ROI
        Point2f find_roi_tl(patch_pt.x - find_radius - kPatchSize / 2.0f,
                            patch_pt.y - find_radius - kPatchSize / 2.0f);
        Point2f find_roi_br(patch_pt.x + find_radius + kPatchSize / 2.0f,
                            patch_pt.y + find_radius + kPatchSize / 2.0f);

        if (find_roi_tl.x < 0 || find_roi_tl.y < 0 ||
            find_roi_br.x >= find_img.cols || find_roi_br.y >= find_img.rows) {
            // If search ROI is outside of this frame, ignore this point
            continue;
        }

        Rect find_roi(find_roi_tl, find_roi_br);

        // Find patch location
        Mat zncc;
        matchTemplate(find_img(find_roi), patch, zncc, TM_CCOEFF_NORMED);

        Point2f found_roi_pt;
        Point roi_max_loc;
        if (countNonZero(zncc) < 1) {
            // Becaue ZNCC is all zero, can not decide match location
            continue;
        }

        minMaxLoc(zncc(Rect(1, 1, zncc.cols - 2, zncc.rows - 2)), NULL, NULL,
                  NULL, &roi_max_loc);
        if (patch_pyramid_level == 0) {
            found_roi_pt = Point2f(roi_max_loc) + Point2f(1, 1);
        } else {
            Point2f sub_roi_max_loc;
            Rect sub_roi(roi_max_loc.x, roi_max_loc.y, 3, 3);

            if (countNonZero(zncc(sub_roi)) < 1) {
                // Becaue ZNCC sub ROI is all zero, can not estimate subpixel
                // location
                continue;
            }

            Util::FindExtrema(zncc(sub_roi), sub_roi_max_loc);

            found_roi_pt = Point2f(roi_max_loc.x + sub_roi_max_loc.x,
                                   roi_max_loc.y + sub_roi_max_loc.y);
        }

        Point2f found_img_pt((found_roi_pt.x - find_radius) + patch_pt.x,
                             (found_roi_pt.y - find_radius) + patch_pt.y);

        found_img_pts.push_back(found_img_pt);
        found_real_obj_pts.push_back(ref_frame->real_obj_pts().at(i));
    }

    if (found_img_pts.size() < 10) {
        return false;
    }

    // Represent match points as 0-level pyramid layer
    Mat(found_img_pts, false) *= pow(2.0, patch_pyramid_level);

    return true;
}
