cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

include(RunCMake)
include("${RunCMake_SOURCE_DIR}/CPackTestHelpers.cmake")

# run_cpack_test args: TEST_NAME "GENERATORS" RUN_CMAKE_BUILD_STEP "PACKAGING_TYPES"
run_cpack_test(CUSTOM_BINARY_SPEC_FILE "RPM" false "MONOLITHIC;COMPONENT")
run_cpack_test(CUSTOM_NAMES "RPM;DEB;TGZ" true "COMPONENT")
run_cpack_test(DEBUGINFO "RPM;DEB" true "COMPONENT")
run_cpack_test_subtests(DEFAULT_PERMISSIONS "CMAKE_var_set;CPACK_var_set;both_set;invalid_CMAKE_var;invalid_CPACK_var" "RPM;DEB" false "MONOLITHIC;COMPONENT")
run_cpack_test(DEPENDENCIES "RPM;DEB" true "COMPONENT")
run_cpack_test(DIST "RPM" false "MONOLITHIC")
run_cpack_test(EMPTY_DIR "RPM;DEB;TGZ" true "MONOLITHIC;COMPONENT")
run_cpack_test(VERSION "RPM;DEB" false "MONOLITHIC;COMPONENT")
run_cpack_test(EXTRA "DEB" false "COMPONENT")
run_cpack_test_subtests(GENERATE_SHLIBS "soversion_not_zero;soversion_zero" "DEB" true "COMPONENT")
run_cpack_test(GENERATE_SHLIBS_LDCONFIG "DEB" true "COMPONENT")
run_cpack_test(INSTALL_SCRIPTS "RPM" false "COMPONENT")
run_cpack_test(LONG_FILENAMES "DEB" false "MONOLITHIC")
run_cpack_test_subtests(MAIN_COMPONENT "invalid;found" "RPM" false "COMPONENT")
run_cpack_test(MINIMAL "RPM;DEB;7Z;TBZ2;TGZ;TXZ;TZ;ZIP;STGZ;External" false "MONOLITHIC;COMPONENT")
run_cpack_test_package_target(MINIMAL "RPM;DEB;7Z;TBZ2;TGZ;TXZ;TZ;ZIP;STGZ;External" false "MONOLITHIC;COMPONENT")
run_cpack_test_subtests(PACKAGE_CHECKSUM "invalid;MD5;SHA1;SHA224;SHA256;SHA384;SHA512" "TGZ" false "MONOLITHIC")
run_cpack_test(PARTIALLY_RELOCATABLE_WARNING "RPM" false "COMPONENT")
run_cpack_test(PER_COMPONENT_FIELDS "RPM;DEB" false "COMPONENT")
run_cpack_test_subtests(SINGLE_DEBUGINFO "no_main_component;one_component;one_component_main;no_debuginfo;one_component_no_debuginfo;no_components;valid" "RPM" true "CUSTOM")
run_cpack_test(EXTRA_SLASH_IN_PATH "RPM" true "COMPONENT")
run_cpack_source_test(SOURCE_PACKAGE "RPM")
run_cpack_test(SUGGESTS "RPM" false "MONOLITHIC")
run_cpack_test(SYMLINKS "RPM;TGZ" false "MONOLITHIC;COMPONENT")
set(ENVIRONMENT "SOURCE_DATE_EPOCH=123456789")
run_cpack_test(TIMESTAMPS "DEB;TGZ" false "COMPONENT")
unset(ENVIRONMENT)
run_cpack_test(USER_FILELIST "RPM" false "MONOLITHIC")
run_cpack_test(MD5SUMS "DEB" false "MONOLITHIC;COMPONENT")
run_cpack_test(CPACK_INSTALL_SCRIPT "ZIP" false "MONOLITHIC")
run_cpack_test(DEB_PACKAGE_VERSION_BACK_COMPATIBILITY "DEB" false "MONOLITHIC;COMPONENT")
run_cpack_test_subtests(EXTERNAL "none;good;good_multi;bad_major;bad_minor;invalid_good;invalid_bad;stage_and_package" "External" false "MONOLITHIC;COMPONENT")
