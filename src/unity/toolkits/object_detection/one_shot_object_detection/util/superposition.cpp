/* Copyright © 2019 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */

#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>

#include <unity/toolkits/object_detection/one_shot_object_detection/util/mapping_function.hpp>
#include <unity/toolkits/object_detection/one_shot_object_detection/util/superposition.hpp>
#include <unity/toolkits/object_detection/one_shot_object_detection/util/quadrilateral_geometry.hpp>

static const boost::gil::rgb8_pixel_t RGB_WHITE(255,255,255);
static const boost::gil::rgba8_pixel_t RGBA_WHITE(255,255,255,0);

namespace turi {
namespace one_shot_object_detection {
namespace data_augmentation {

void superimpose_image(const boost::gil::rgb8_image_t::view_t &masked,
                       const boost::gil::rgba8_image_t::view_t &transformed,
                       const boost::gil::rgba8_image_t::view_t &background) {
  boost::gil::transform_pixels(transformed, background, masked, 
    [](boost::gil::rgba8_ref_t a, boost::gil::rgba8_ref_t b) {
          boost::gil::red_t   R;
          boost::gil::green_t G;
          boost::gil::blue_t  B;
          boost::gil::alpha_t A;
          auto AoverB = [](size_t C_a, size_t C_b, size_t alpha_a, size_t alpha_b) {
            return ((C_a * alpha_a + C_b * alpha_b * (1 - alpha_a))/(
                      alpha_a + alpha_b * (1 - alpha_a)));
          };
          return  boost::gil::rgb8_image_t::value_type (
                    AoverB(get_color(a,R), get_color(b,R), get_color(a,A)/255, get_color(b,A)/255),
                    AoverB(get_color(a,G), get_color(b,G), get_color(a,A)/255, get_color(b,A)/255),
                    AoverB(get_color(a,B), get_color(b,B), get_color(a,A)/255, get_color(b,A)/255)
          );
  });
}

void superimpose_rgb_object_image(ParameterSampler &parameter_sampler,
                                  const flex_image &object,
                                  const boost::gil::rgb8_image_t::view_t &masked,
                                  const boost::gil::rgba8_image_t::view_t &transformed,
                                  const boost::gil::rgba8_image_t::view_t &background) {
  Eigen::Matrix<float, 3, 3> M = parameter_sampler.get_transform().inverse();
  boost::gil::rgb8_image_t::view_t starter_image_view = interleaved_view(
    object.m_width,
    object.m_height,
    (boost::gil::rgb8_pixel_t*) (object.get_image_data()),
    object.m_channels * object.m_width // row length in bytes
  );
  boost::gil::rgba8_image_t starter_image_rgba(boost::gil::rgba8_image_t::point_t(starter_image_view.dimensions()));
  // The following line uses the color_convert conversion to convert the 
  // background from RGB to RGBA
  boost::gil::copy_and_convert_pixels(
    starter_image_view,
    view(starter_image_rgba)
  );
  resample_pixels(view(starter_image_rgba), transformed, M, boost::gil::bilinear_sampler());
  color_quadrilateral(transformed, parameter_sampler.get_warped_corners());
  superimpose_image(masked, transformed, background);
}

void superimpose_rgba_object_image(ParameterSampler &parameter_sampler,
                                   const flex_image &object,
                                   const boost::gil::rgb8_image_t::view_t &masked,
                                   const boost::gil::rgba8_image_t::view_t &transformed,
                                   const boost::gil::rgba8_image_t::view_t &background) {
  Eigen::Matrix<float, 3, 3> M = parameter_sampler.get_transform().inverse();
  boost::gil::rgba8_image_t::view_t starter_image_view = interleaved_view(
    object.m_width,
    object.m_height,
    (boost::gil::rgba8_pixel_t*) (object.get_image_data()),
    object.m_channels * object.m_width // row length in bytes
  );
  resample_pixels(starter_image_view, transformed, M, boost::gil::bilinear_sampler());
  superimpose_image(masked, transformed, background);
}

flex_image create_synthetic_image(const boost::gil::rgb8_image_t::view_t &background_view,
                                  ParameterSampler &parameter_sampler,
                                  const flex_image &object) {
  boost::gil::rgba8_image_t background_rgba(boost::gil::rgba8_image_t::point_t(background_view.dimensions()));
  boost::gil::copy_and_convert_pixels(
    background_view,
    view(background_rgba)
  );
  boost::gil::rgba8_image_t transformed(boost::gil::rgba8_image_t::point_t(background_view.dimensions()));
  fill_pixels(view(transformed), RGBA_WHITE);
  boost::gil::rgb8_image_t masked(boost::gil::rgba8_image_t::point_t(background_view.dimensions()));
  fill_pixels(view(masked), RGB_WHITE);
  if (object.m_channels == 4) {
    superimpose_rgba_object_image(parameter_sampler, object, view(masked), view(transformed), view(background_rgba));
  } else {
    superimpose_rgb_object_image(parameter_sampler, object, view(masked), view(transformed), view(background_rgba));
  }
  return flex_image(masked);
}

} // data_augmentation
} // one_shot_object_detection
} // turi
