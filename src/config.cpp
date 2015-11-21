#include "config.hpp"

Config::Config(const Size& cam_img_size, const Matx33d& int_mat)
    : cam_img_size_(cam_img_size), int_mat_(int_mat) {}

Config::~Config() {}

const Config Config::Get() {
    Size size = Size(640, 480);
    Matx33d int_mat(500, 0, size.width / 2.0, 0, 500, size.height / 2.0, 0, 0,
                    1);

    return Config(size, int_mat);
}

const Size& Config::cam_img_size() const { return cam_img_size_; }

const Matx33d& Config::int_mat() const { return int_mat_; }
