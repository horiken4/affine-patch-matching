#ifndef PATCH_FINDER_HPP
#define PATCH_FINDER_HPP

#include <opencv2/opencv.hpp>

#include "pose.hpp"
#include "frame.hpp"
#include "config.hpp"
#include "util.hpp"
#include "framefactory.hpp"

using namespace std;
using namespace cv;

class PatchFinder {
   public:
    static const int kPatchSize;
    static const int kCoarsePatchSearchRadius;
    static const int kFinePatchSearchRadius;
    static const int kNumCoarseMatches;
    static const int kNumFineMatches;

   public:
    PatchFinder(const Matx33d& int_mat);
    ~PatchFinder();

    bool EstimatePose(const shared_ptr<Frame>& ref_frame,
                      const shared_ptr<Frame>& target_frame,
                      const Pose& prior_pose, Pose& pose);
    bool EstimatePoseCoarseToFine(const shared_ptr<Frame>& ref_frame,
                                  const shared_ptr<Frame>& target_frame,
                                  const Pose& prior_pose, Pose& pose);

    void MakePatch(const shared_ptr<ReferenceFrame>& ref_frame,
                   const Pose& prior_pose, vector<int>& patch_pyramid_levels,
                   vector<Point2f>& patch_pts, vector<Mat>& patches);
    bool FindPatch(const shared_ptr<ReferenceFrame> ref_frame,
                   const shared_ptr<Frame>& target_frame,
                   const vector<int>& patch_pyramid_levels,
                   const vector<Point2f>& patch_pts, const vector<Mat>& patches,
                   const vector<int>& used_patch_indices, int find_radius,
                   vector<Point3f>& found_real_obj_pts,
                   vector<Point2f>& found_img_pts);

   private:
    Matx33d int_mat_;
};

#endif
