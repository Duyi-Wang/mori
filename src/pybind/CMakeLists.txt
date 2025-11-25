find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

add_library(mori_pybinds SHARED mori.cpp pybind.cpp)

# Use bundled dlpack headers
set(DLPACK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/dlpack/include)
message(STATUS "Using DLPack from: ${DLPACK_INCLUDE_DIR}")

target_include_directories(
  mori_pybinds PUBLIC ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS}
                      ${DLPACK_INCLUDE_DIR})
target_link_libraries(
  mori_pybinds
  mori_ops
  mori_io
  Python3::Python
  pybind11::module
  hip::host
  hip::device)

# For python packages to find dependent libraries
set_target_properties(
  mori_pybinds
  PROPERTIES BUILD_RPATH "$ORIGIN"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE)
