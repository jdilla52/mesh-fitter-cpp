#
# Create a target for executing the Catch unit tests.
#

set(CMAKE_CXX_FLAGS "-std=c++17 -fPIC")
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

add_executable(3dMethodsTest
        main.cpp
        3dMethods.cpp
        )

target_include_directories(3dMethodsTest
        PUBLIC
        test/
        "${PROJECT_BINARY_DIR}"
        )

target_link_libraries(3dMethodsTest
        Catch2::Catch2
        3dMethods
        Eigen3::Eigen
        igl::core
        )

#target_link_libraries(${PROJECT_NAME} PUBLIC igl::core igl::opengl_glfw)
cmake_policy(SET CMP0110 NEW)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")
Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v2.13.1)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)

# Load and use the .cmake file provided by Catch so all the test cases
# are made available to CTest.
include(ParseAndAddCatchTests)
ParseAndAddCatchTests(3dMethodsTest)