#include <unity/toolkits/one_shot_object_detection/one_shot_object_detector.hpp>

#include <unity/toolkits/object_detection/object_detector.hpp>

// TODO: Clean up imports.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>
#include <random>

#include <logger/assertions.hpp>
#include <logger/logger.hpp>
#include <random/random.hpp>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/jpeg.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>
#include <boost/gil/utilities.hpp>

#include <unity/lib/gl_sframe.hpp>

#include <image/numeric_extension/perspective_projection.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

using turi::coreml::MLModelWrapper;
using namespace Eigen;
using namespace boost::gil;

namespace turi {
namespace one_shot_object_detection {

class ParameterSweep {
public:
  ParameterSweep(int width, int height) {
    width_ = width;
    height_ = height;
  }

  double deg_to_rad(double angle) {
    return angle * M_PI / 180.0;
  }

  double get_theta() {
    return deg_to_rad(theta_);
  }

  double get_phi() {
    return deg_to_rad(phi_);
  }

  double get_gamma() {
    return deg_to_rad(gamma_);
  }

  int get_dz() {
    return dz_;
  }

  double get_focal() {
    return focal_;
  }

  void sample(long seed) {
    double theta_mean, phi_mean, gamma_mean;
    std::srand(seed);
    theta_mean = theta_means_[std::rand() % theta_means_.size()];
    std::srand(seed+1);
    phi_mean = phi_means_[std::rand() % phi_means_.size()];
    std::srand(seed+2);
    gamma_mean = gamma_means_[std::rand() % gamma_means_.size()];
    std::normal_distribution<double> theta_distribution(theta_mean, angle_stdev_);
    std::normal_distribution<double> phi_distribution(phi_mean, angle_stdev_);
    std::normal_distribution<double> gamma_distribution(gamma_mean, angle_stdev_);
    std::normal_distribution<double> focal_distribution((double)width_, focal_stdev_);
    theta_generator_.seed(seed+3);
    theta_ = theta_distribution(theta_generator_);
    phi_generator_.seed(seed+4);
    phi_ = phi_distribution(phi_generator_);
    gamma_generator_.seed(seed+5);
    gamma_ = gamma_distribution(gamma_generator_);
    focal_generator_.seed(seed+6);
    focal_ = focal_distribution(focal_generator_);
    std::uniform_int_distribution<int> dz_distribution(
      std::max(width_, height_), max_depth_);
    dz_generator_.seed(seed+7);
    dz_ = focal_ + dz_distribution(dz_generator_);
  }

private:
  int width_;
  int height_;
  int max_depth_ = 13000;
  double angle_stdev_ = 20.0;
  double focal_stdev_ = 40.0;
  std::vector<double> theta_means_ = {-180.0, 0.0, 180.0};
  std::vector<double> phi_means_   = {-180.0, 0.0, 180.0};
  std::vector<double> gamma_means_ = {-180.0, -90.0, 0.0, 90.0, 180.0};
  std::default_random_engine theta_generator_;
  std::default_random_engine phi_generator_;
  std::default_random_engine gamma_generator_;
  std::default_random_engine dz_generator_;
  std::default_random_engine focal_generator_;
  double theta_;
  double phi_;
  double gamma_;
  int dz_;
  double focal_;

};

gl_sframe _augment_data(gl_sframe data, gl_sframe backgrounds, long seed) {
  // TODO: Get input image from the data sframe.
  // TODO: Use backgrounds from the background SFrame.
  // TODO: Generalize 1024 and 676 to be the width and height of the image 
  //       passed in.
  ParameterSweep parameter_sampler = ParameterSweep(2*1024, 2*676);
  // TODO: Take n as input.
  int n = 1;
  std::vector<flexible_type> annotations;
  std::vector<flexible_type> images;
  for (int i = 0; i < n; i++) {
    parameter_sampler.sample(seed+i);

    Matrix3f mat = get_transformation_matrix(2*1024, 2*676,
      parameter_sampler.get_theta(),
      parameter_sampler.get_phi(),
      parameter_sampler.get_gamma(),
      1024/2,
      676/2,
      parameter_sampler.get_dz(),
      parameter_sampler.get_focal());
    int original_top_left_x = 0;
    int original_top_left_y = 0;
    int original_top_right_x = 1024;
    int original_top_right_y = 0;
    int original_bottom_left_x = 0;
    int original_bottom_left_y = 676;
    int original_bottom_right_x = 1024;
    int original_bottom_right_y = 676;

    Vector3f top_left_corner(3)   , top_right_corner(3);
    Vector3f bottom_left_corner(3), bottom_right_corner(3);
    top_left_corner     << original_top_left_x    , original_top_left_y    , 1;
    top_right_corner    << original_top_right_x   , original_top_right_y   , 1;
    bottom_left_corner  << original_bottom_left_x , original_bottom_left_y , 1;
    bottom_right_corner << original_bottom_right_x, original_bottom_right_y, 1;

    std::vector<Vector3f> warped_corners = {mat * top_left_corner   ,
                                            mat * top_right_corner  ,
                                            mat * bottom_left_corner,
                                            mat * bottom_right_corner};

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float min_y = std::numeric_limits<float>::max(); 
    float max_y = std::numeric_limits<float>::min();
    for (auto corner: warped_corners) {
      corner[0] /= corner[2];
      corner[1] /= corner[2];
      min_x = std::min(min_x, corner[0]);
      max_x = std::max(max_x, corner[0]);
      min_y = std::min(min_y, corner[1]);
      max_y = std::max(max_y, corner[1]);
    }
    float center_x = (min_x + max_x) / 2;
    float center_y = (min_y + max_y) / 2;
    float bounding_box_width  = max_x - min_x;
    float bounding_box_height = max_y - min_y;

    flex_dict coordinates = {std::make_pair("x", center_x),
                             std::make_pair("y", center_y),
                             std::make_pair("width", bounding_box_width),
                             std::make_pair("height", bounding_box_height)
                            };
    flex_dict annotation = {std::make_pair("coordinates", coordinates),
                            std::make_pair("label", "placeholder")
                           };

    rgb8_image_t starter_image, background;
    // TODO: Don't hardcode this.
    read_image("in-affine.jpg", starter_image, jpeg_tag());
    // TODO: Don't hardcode this. Fetch this from the backgrounds SFrame in a 
    // loop.
    read_image("background.jpg", background, jpeg_tag());

    matrix3x3<double> M(mat.inverse());
    // TODO: Use this mask during superposition on a random background.
    rgb8_image_t mask(rgb8_image_t::point_t(view(background).dimensions()));
    fill_pixels(view(mask),rgb8_pixel_t(0, 0, 0));
    rgb8_image_t transformed(rgb8_image_t::point_t(view(starter_image).dimensions()*2));
    fill_pixels(view(transformed),rgb8_pixel_t(255, 255, 255));
    resample_pixels(const_view(starter_image), view(transformed), M, bilinear_sampler());

    // TODO: Write these images into an SFrame that this function can return.
    std::string output_filename = "out-affine-" + std::to_string(i) + ".jpg";
    write_view(output_filename, view(transformed), jpeg_tag());

    annotations.push_back(annotation);
    // TODO: figure out how to convert rgb8_image_t to flex_image.
  }
  const std::map<std::string, std::vector<flexible_type> >& augmented_data = {{"annotations", annotations}};
  gl_sframe augmented_data_out = gl_sframe(augmented_data);
  // TODO: Return the augmented data once the SFrame is written.
  return augmented_data_out;
}

one_shot_object_detector::one_shot_object_detector() {
  model_.reset(new turi::object_detection::object_detector());
}

gl_sframe one_shot_object_detector::augment(gl_sframe data,
                                            std::string target_column_name,
                                            gl_sframe backgrounds,
                                            long seed){
  gl_sframe augmented_data = _augment_data(data, backgrounds, seed);
  // TODO: Call object_detector::train from here once we incorporate mxnet into
  // the C++ Object Detector.
  return augmented_data;
}
/* TODO: We probably don't need `evaluate` and `export_to_coreml` on the C++ 
         side for now, but it may not hurt to leave it here.
 */
variant_map_type one_shot_object_detector::evaluate(gl_sframe data, 
  std::string metric, std::map<std::string, flexible_type> options) {
  return model_->evaluate(data, metric, options);
}

std::shared_ptr<MLModelWrapper> one_shot_object_detector::export_to_coreml(
  std::string filename, std::map<std::string, flexible_type> options) {
  return model_->export_to_coreml(filename, options);
}

} // one_shot_object_detection
} // turi