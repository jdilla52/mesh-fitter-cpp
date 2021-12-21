option(LIBIGL_WITH_GLFW "Build target igl::glfw" ON)
include(libigl)
add_library(3dMethods STATIC
        pca.cpp
        icp.h
        )
target_link_libraries(3dMethods PRIVATE Eigen3::Eigen igl::core)

target_include_directories(3dMethods PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_BINARY_DIR}" "${PROJECT_BINARY_DIR}")
target_include_directories(3dMethods PUBLIC src/lambdaRunner)

