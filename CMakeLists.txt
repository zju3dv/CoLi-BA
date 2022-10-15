cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(coli)

if(POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY) 
endif()

################################################################################
# Find packages
################################################################################

find_package(Eigen3 REQUIRED)

find_package(g2o REQUIRED)

find_package(Ceres REQUIRED)

################################################################################
# Compiler specific configuration
################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -w")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")

################################################################################
# Add sources
################################################################################
 
include_directories(
        ${PROJECT_SOURCE_DIR}/src
        ${CERES_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/ceres-solver/internal/
        /usr/include/suitesparse
)

set(COLI_SOURCE_FILES
        src/optim/linear_solving.cc
        src/optim/linear_solver.cc
        src/optim/linear_solver_coli.cc
        src/optim/lm_minimizer.cc
        src/optim/local_parameterization.cc

        src/run_ba_coli.cc
        src/run_ba_ceres.cc
        src/run_ba_g2o.cc
        )

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_library(
        ${PROJECT_NAME}
        ${COLI_SOURCE_FILES}
)

target_link_libraries(
        ${PROJECT_NAME}
        ${CERES_LIBRARIES}
        Eigen3::Eigen
        -ltbb
        g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
)

add_executable(main src/main.cc)
target_link_libraries(main ${PROJECT_NAME})
