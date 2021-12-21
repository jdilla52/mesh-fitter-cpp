cmake_minimum_required(VERSION 3.3)
set(CMAKE_CXX_STANDARD 17)
project(3dMethods LANGUAGES CXX)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

set(LAMBDA_CONFIG_HEADER ${CMAKE_BINARY_DIR}/methods_config.h)
message("Generating ${LAMBDA_CONFIG_HEADER}...")
configure_file(src/3dMethods/methods_config.in ${LAMBDA_CONFIG_HEADER} )

# library
add_subdirectory(src/3dMethods)
target_include_directories(3dMethods PUBLIC src)
add_subdirectory(test)
# tests
if($ENV{TEST_3DMETHODS})
    message("Generating tests")
    add_subdirectory(test)
elseif(${TEST_3DMETHODS})
    message("Generating tests")
    add_subdirectory(test)
else()
    message("Building without tests")
endif()