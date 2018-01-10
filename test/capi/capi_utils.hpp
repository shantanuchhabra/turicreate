#ifndef CAPI_TEST_UTILS
#define CAPI_TEST_UTILS

#include <capi/TuriCore.h>
#include <vector>

static tc_flex_list* make_flex_list_double(const std::vector<double>& v) { 

  tc_error* error = NULL; 
  
  tc_flex_list* fl = tc_flex_list_create(&error);

  TS_ASSERT(error == NULL); 

  size_t pos = 0; 

  for(double t : v) { 
    tc_flexible_type* ft = tc_ft_create_from_double(t, &error);

    TS_ASSERT(error == NULL); 

    size_t idx = tc_flex_list_add_element(fl, ft, &error);

    TS_ASSERT(error == NULL); 

    TS_ASSERT_EQUALS(idx, pos); 
    ++pos;   

    size_t actual_size = tc_flex_list_size(fl); 

    TS_ASSERT_EQUALS(pos, actual_size); 

    tc_ft_destroy(ft);  
  }

  // Go through and make sure things are equal
  for(size_t i = 0; i < v.size(); ++i) { 
    tc_flexible_type* ft = tc_flex_list_extract_element(fl, i, &error); 

    TS_ASSERT(error == NULL); 

    TS_ASSERT(tc_ft_is_double(ft) != 0);

    double val = tc_ft_double(ft, &error); 
    TS_ASSERT(error == NULL); 

    TS_ASSERT(v[i] == val); 
      
    tc_ft_destroy(ft);
  }

  return fl;
} 

static tc_sarray* make_sarray_double(const std::vector<double>& v) { 

  tc_error* error = NULL; 
  
  tc_flex_list* fl = make_flex_list_double(v); 

  tc_sarray* sa = tc_sarray_create_from_list(fl, &error);

  TS_ASSERT(error == NULL); 

  { 
    // Make sure it gets out what we want it to.
    for(size_t i = 0; i < v.size(); ++i) { 
      tc_flexible_type* ft = tc_sarray_extract_element(sa, i, &error); 
      TS_ASSERT(error == NULL); 

      TS_ASSERT(tc_ft_is_double(ft) != 0);

      double val = tc_ft_double(ft, &error); 
      TS_ASSERT(error == NULL); 

      TS_ASSERT(v[i] == val); 

      tc_ft_destroy(ft);
    }
  }

  return sa;
}


static tc_sframe* make_sframe_double(const std::vector<std::pair<std::string, std::vector<double> > >& data ) { 

    tc_error* error = NULL; 

    tc_sframe* sf = tc_sframe_create_empty(&error); 
      
    TS_ASSERT(error == NULL); 

    for(auto p : data) { 

      tc_sarray* sa = make_sarray_double(p.second); 

      tc_sframe_add_column(sf, p.first.c_str(), sa, &error); 

      TS_ASSERT(error == NULL); 

      tc_sarray_destroy(sa); 
    }

    // Check everything 
    for(auto p : data) { 
      // Make sure it gets out what we want it to.
      tc_sarray* sa = tc_sframe_extract_column_by_name(sf, p.first.c_str(), &error); 

      TS_ASSERT(error == NULL); 

      tc_sarray* ref_sa = make_sarray_double(p.second);

      TS_ASSERT(tc_sarray_equals(sa, ref_sa, &error));     

      TS_ASSERT(error == NULL); 

      tc_sarray_destroy(sa); 
    }

    return sf; 
  }



#endif
