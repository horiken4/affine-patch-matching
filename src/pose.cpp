#include "pose.hpp"

Pose::Pose() {}
Pose::Pose(const Vec3d& rvec, const Vec3d& tvec) : rvec_(rvec), tvec_(tvec) {
    UpdateMat();
}
Pose::~Pose() {}

const Vec3d& Pose::rvec() const { return rvec_; }
const Vec3d& Pose::tvec() const { return tvec_; }
const Matx34d& Pose::mat() const { return mat_; }

void Pose::set_rvec(const Vec3d& rvec) {
    rvec_ = rvec;
    UpdateMat();
}
void Pose::set_tvec(const Vec3d& tvec) {
    tvec_ = tvec;
    UpdateMat();
}

void Pose::UpdateMat() {
    Matx33d rmat;
    Rodrigues(rvec_, rmat);

    mat_(0, 0) = rmat(0, 0);
    mat_(0, 1) = rmat(0, 1);
    mat_(0, 2) = rmat(0, 2);
    mat_(1, 0) = rmat(1, 0);
    mat_(1, 1) = rmat(1, 1);
    mat_(1, 2) = rmat(1, 2);
    mat_(2, 0) = rmat(2, 0);
    mat_(2, 1) = rmat(2, 1);
    mat_(2, 2) = rmat(2, 2);

    mat_(0, 3) = tvec_(0);
    mat_(1, 3) = tvec_(1);
    mat_(2, 3) = tvec_(2);
}
