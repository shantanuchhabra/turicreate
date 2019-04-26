/* Copyright © 2017 Apple Inc. All rights reserved.
 *
 * Use of this source code is governed by a BSD-3-clause license that can
 * be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
 */
#include <image/image_type.hpp>

namespace turi{

image_type::image_type(boost::gil::rgb8_image_t gil_image) {
  // typedef bit_aligned_image3_type<1,1,1, boost::gil::rgb8_layout_t>::type image_t;
  auto it = view(gil_image).begin();
  auto data = (uint8_t*) &boost::gil::at_c<0>(*it);
  boost::gil::rgb8_image_t::const_view_t view = const_view(gil_image);
  // const char* buffer = view.begin().x();
  m_height = gil_image.height();
  m_width = gil_image.width();
  m_channels = boost::gil::num_channels<boost::gil::rgb8_image_t>();
  m_image_data_size = m_height * m_width * m_channels;
  m_version = IMAGE_TYPE_CURRENT_VERSION;
  m_format = Format::RAW_ARRAY;
  m_image_data.reset(new char[m_image_data_size]);
  memcpy(&m_image_data[0], data, m_image_data_size);
}

image_type::image_type(const char* image_data, size_t height, size_t width, size_t channels, size_t image_data_size, int version, int format){
  m_image_data.reset(new char[image_data_size]);
  memcpy(&m_image_data[0], image_data, image_data_size);
  m_height = height;
  m_width = width;
  m_channels = channels;
  m_image_data_size = image_data_size;
  m_version = (char)version;
  m_format = static_cast<Format>(format);
}


void image_type::save(oarchive& oarc) const {
  oarc << m_version << m_height << m_width << m_channels << m_format <<m_image_data_size;
  if (m_image_data_size > 0) {
      oarc.write(&m_image_data[0], m_image_data_size);
  }
}

void image_type::load(iarchive& iarc) {
  iarc >> m_version >> m_height >> m_width >> m_channels >> m_format >> m_image_data_size;
  if (m_image_data_size > 0){
      m_image_data.reset (new char[m_image_data_size]);
      iarc.read(&m_image_data[0], m_image_data_size);
  } else {
      m_image_data.reset();
  }
}

const unsigned char* image_type::get_image_data() const { 
  if (m_image_data_size > 0){
    return (const unsigned char*)&m_image_data[0];
  } else{
    return NULL;
  }
}

}
