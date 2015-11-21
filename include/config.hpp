#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <opencv2/opencv.hpp>

using namespace cv;

class Config {
   private:
    Config();
    Config(const Size& cam_img_size, const Matx33d& int_mat);

   public:
    ~Config();
    static const Config Get();
    const Size& cam_img_size() const;
    const Matx33d& int_mat() const;

   private:
    // Camera image size for synthesize camera image
    Size cam_img_size_;

    // Intrinsic matrix
    Matx33d int_mat_;
};
#endif
