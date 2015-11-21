#ifndef POSE_HPP
#define POSE_HPP

#include <opencv2/opencv.hpp>

using namespace cv;

class Pose {
   public:
    Pose();
    Pose(const Vec3d& rvec, const Vec3d& tvec);
    virtual ~Pose();

    const Vec3d& rvec() const;
    const Vec3d& tvec() const;
    const Matx34d& mat() const;

    void set_rvec(const Vec3d& rvec);
    void set_tvec(const Vec3d& tvec);

    private:
    void UpdateMat();

   private:
    Vec3d rvec_;
    Vec3d tvec_;
    Matx34d mat_;
};

#endif
