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
#include <boost/gil/extension/toolbox/metafunctions.hpp>
#include <boost/gil/extension/toolbox/metafunctions/gil_extensions.hpp>

#include <unity/lib/gl_sframe.hpp>

#include <image/numeric_extension/perspective_projection.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

// This is an important import, please note.
#include <image/image_util_impl.hpp>

#define BLACK rgb8_pixel_t(0,0,0)
#define WHITE rgb8_pixel_t(255,255,255)

using turi::coreml::MLModelWrapper;
using namespace Eigen;
using namespace boost::gil;

namespace turi {
namespace one_shot_object_detection {

class ParameterSweep {
public:
  ParameterSweep(int width, int height, int dx, int dy) {
    width_ = width;
    height_ = height;
    dx_ = dx;
    dy_ = dy;
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

  Matrix3f get_transform() {
    return transform_;
  }

  void set_warped_corners(const std::vector<Vector3f> &warped_corners) {
    warped_corners_ = warped_corners;
  }

  std::vector<Vector3f> get_warped_corners() {
    return warped_corners_;
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
    transform_ = get_transformation_matrix(
      width_, height_, theta_, phi_, gamma_, dx_, dy_, dz_, focal_);
    warped_corners_.reserve(4);
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
  int dx_;
  int dy_;
  int dz_;
  double focal_;
  Matrix3f transform_;
  std::vector<Vector3f> warped_corners_;
};

namespace {

class Line {
public:
  Line(Vector3f P1, Vector3f P2) {
    float x1 = P1[0], y1 = P1[1];
    float x2 = P2[0], y2 = P2[1];
    m_a = (y2 - y1) / (x2 - x1);
    m_b = -1;
    m_c = y1 - (x1 * (y2 - y1) / (x2 - x1));
  }

  bool side_of_line(int x, int y) {
    return (m_a*x + m_b*y + m_c > 0);
  }

private:
  float m_a, m_b, m_c; // ax + by + c = 0
};

bool is_in_quadrilateral(int x, int y, std::vector<Vector3f> warped_corners) {
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::min();
  for (auto corner: warped_corners) {
    min_x = std::min(min_x, corner[0]);
    max_x = std::max(max_x, corner[0]);
    min_y = std::min(min_y, corner[1]);
    max_y = std::max(max_y, corner[1]);
  }
  if (x < min_x || x > max_x || y < min_y || y > max_y) {
    return false;
  }
  // swap last two entries to make the corners cyclic.
  Vector3f temp = warped_corners[2];
  warped_corners[2] = warped_corners[3];
  warped_corners[3] = temp;
  int num_true = 0;
  for (unsigned int index = 0; index < warped_corners.size(); index++) {
    auto left_corner = warped_corners[index % warped_corners.size()];
    auto right_corner = warped_corners[(index+1) % warped_corners.size()];
    Line L = Line(left_corner, right_corner);
    num_true += (L.side_of_line(x, y));
  }
  return (num_true == 2);
}

}

void color_quadrilateral(const rgb8_image_t::view_t &mask_view, 
                         const rgb8_image_t::view_t &mask_complement_view, 
                         std::vector<Vector3f> warped_corners) {
  // printf("mask_view height = %ld, width = %ld\n", mask_view.height(), mask_view.width());
  for (int y = 0; y < mask_view.height(); ++y) {
    auto mask_row_iterator = mask_view.row_begin(y);
    auto mask_complement_row_iterator = mask_complement_view.row_begin(y);
    for (int x = 0; x < mask_view.width(); ++x) {
      if (is_in_quadrilateral(x, y, warped_corners)) {
        mask_row_iterator[x] = WHITE;
        mask_complement_row_iterator[x] = BLACK;
      }
    }
  }
}

void superimpose_image(const rgb8_image_t::view_t &masked,
                       const rgb8_image_t::view_t &mask,
                       const rgb8_image_t::view_t &transformed,
                       const rgb8_image_t::view_t &mask_complement,
                       const rgb8_image_t::view_t &background) {
  for (int y = 0; y < masked.height(); ++y) {
    // printf("y=%d\n", y);
    auto masked_row_it = masked.row_begin(y);
    auto mask_row_it = mask.row_begin(y);
    auto transformed_row_it = transformed.row_begin(y);
    auto mask_complement_row_it = mask_complement.row_begin(y);
    auto background_row_it = background.row_begin(y);
    for (int x = 0; x < masked.width(); ++x) {
      // printf("\tx=%d\n", x);
      masked_row_it[x][0] = (mask_row_it[x][0]/255 * transformed_row_it[x][0] + 
        mask_complement_row_it[x][0]/255 * background_row_it[x][0]);
      masked_row_it[x][1] = (mask_row_it[x][1]/255 * transformed_row_it[x][1] + 
        mask_complement_row_it[x][1]/255 * background_row_it[x][1]);
      // printf("\tshould be ");
      masked_row_it[x][2] = (mask_row_it[x][2]/255 * transformed_row_it[x][2] + 
        mask_complement_row_it[x][2]/255 * background_row_it[x][2]);
      // printf("\tpixel value=%d\n", masked_row_it[x][2]);
    }
  }
}

static std::map<std::string,size_t> generate_column_index_map(
    const std::vector<std::string>& column_names) {
    std::map<std::string,size_t> index_map;
    for (size_t k=0; k < column_names.size(); ++k) {
        index_map[column_names[k]] = k;
    }
    return index_map;
}

flex_dict build_annotation( ParameterSweep &parameter_sampler,
                            int object_width,
                            int object_height,
                            long seed) {

  parameter_sampler.sample(seed);

  int original_top_left_x = 0;
  int original_top_left_y = 0;
  int original_top_right_x = object_width;
  int original_top_right_y = 0;
  int original_bottom_left_x = 0;
  int original_bottom_left_y = object_height;
  int original_bottom_right_x = object_width;
  int original_bottom_right_y = object_height;

  Vector3f top_left_corner(3)   , top_right_corner(3);
  Vector3f bottom_left_corner(3), bottom_right_corner(3);
  top_left_corner     << original_top_left_x    , original_top_left_y    , 1;
  top_right_corner    << original_top_right_x   , original_top_right_y   , 1;
  bottom_left_corner  << original_bottom_left_x , original_bottom_left_y , 1;
  bottom_right_corner << original_bottom_right_x, original_bottom_right_y, 1;

  auto normalize = [](Vector3f corner) {
    corner[0] /= corner[2];
    corner[1] /= corner[2];
    corner[2] = 1.0;
    return corner;
  };
  
  Matrix3f mat = parameter_sampler.get_transform();

  std::vector<Vector3f> warped_corners = {normalize(mat * top_left_corner)   ,
                                          normalize(mat * top_right_corner)  ,
                                          normalize(mat * bottom_left_corner),
                                          normalize(mat * bottom_right_corner)
                                         };
  parameter_sampler.set_warped_corners(warped_corners);

  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::min();
  for (auto corner: warped_corners) {
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
  return annotation;
}

gl_sframe _augment_data(gl_sframe data,
                        std::string image_column_name,
                        std::string target_column_name,
                        gl_sarray backgrounds,
                        long seed) {
  auto column_index_map = generate_column_index_map(data.column_names());
  std::vector<flexible_type> annotations, images;
  for (const auto& row: data.range_iterator()) {
    flex_image object = row[column_index_map[image_column_name]].to<flex_image>();
    std::string label = row[column_index_map[target_column_name]].to<flex_string>();
    int object_width = object.m_width;
    int object_height = object.m_height;
    if (!(object.is_decoded())) {
      decode_image_inplace(object);
    }
    int row_number = -1;
    for (const auto& background_ft: backgrounds.range_iterator()) {
      row_number++;
      flex_image flex_background = background_ft.to<flex_image>();
      if (!(flex_background.is_decoded())) {
        decode_image_inplace(flex_background);
      }
      int background_width = flex_background.m_width;
      int background_height = flex_background.m_height;
      
      ParameterSweep parameter_sampler = ParameterSweep(background_width, 
                                                        background_height,
                                                        (background_width-object_width)/2,
                                                        (background_height-object_height)/2);

      flex_dict annotation = build_annotation(parameter_sampler, 
                                              object_width, object_height, 
                                              seed+row_number);

      std::vector<Vector3f> corners = parameter_sampler.get_warped_corners();
      
      // create a gil view of the src buffer
      rgb8_image_t::view_t starter_image_view = interleaved_view(
        object_width,
        object_height,
        (rgb8_pixel_t*) (object.get_image_data()),
        3 * object_width // row length in bytes
        );
      
      DASSERT_EQ(flex_background.m_channels, 3);
      DASSERT_TRUE(flex_background.get_image_data() != nullptr);
      DASSERT_TRUE(object.is_decoded());
      DASSERT_TRUE(flex_background.is_decoded());

      rgb8_image_t::view_t background_view = interleaved_view(
        background_width,
        background_height,
        (rgb8_pixel_t*)(flex_background.get_image_data()),
        3 * background_width // row length in bytes
        );
      
      matrix3x3<double> M(parameter_sampler.get_transform().inverse());
      rgb8_image_t mask(rgb8_image_t::point_t(background_view.dimensions()));
      rgb8_image_t mask_complement(rgb8_image_t::point_t(background_view.dimensions()));
      // mask_complement = 1 - mask
      fill_pixels(view(mask), BLACK);
      fill_pixels(view(mask_complement), WHITE);
      color_quadrilateral(view(mask), view(mask_complement), 
        parameter_sampler.get_warped_corners());
      
      rgb8_image_t transformed(rgb8_image_t::point_t(background_view.dimensions()));
      fill_pixels(view(transformed), WHITE);
      resample_pixels(starter_image_view, view(transformed), M, bilinear_sampler());
      
      rgb8_image_t masked(rgb8_image_t::point_t(background_view.dimensions()));
      fill_pixels(view(masked), WHITE);
      // Superposition:
      // mask * warped + (1-mask) * background
      superimpose_image(view(masked), view(mask), view(transformed), 
                                      view(mask_complement), background_view);
      annotations.push_back(annotation);
      images.push_back(flex_image(masked));
    }
  }

  const std::map<std::string, std::vector<flexible_type> >& augmented_data = {
    {"annotation", annotations},
    {"image", images}
  };
  gl_sframe augmented_data_out = gl_sframe(augmented_data);
  return augmented_data_out;
}

one_shot_object_detector::one_shot_object_detector() {
  model_.reset(new turi::object_detection::object_detector());
}

gl_sframe one_shot_object_detector::augment(gl_sframe data,
                                            std::string target_column_name,
                                            gl_sarray backgrounds,
                                            long seed){
  // TODO: Automatically infer the image column name, or throw error if you can't
  // This should just happen on the Python side.
  std::string image_column_name = "image";
  gl_sframe augmented_data = _augment_data(data, image_column_name, target_column_name, backgrounds, seed);
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