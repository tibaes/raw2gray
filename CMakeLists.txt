cmake_minimum_required(VERSION 3.5)

set(VERSION "1.0.0")
set(TARGET_NAME "raw2gray")
project("Imaging tool for converting Raw (Bayer) to RGB" VERSION ${VERSION} LANGUAGES CXX)

include(CMakeToolsHelpers OPTIONAL)

find_package(OpenCV REQUIRED core imgproc imgcodecs highgui videoio)

ADD_EXECUTABLE(${TARGET_NAME} raw2gray.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES
  CXX_STANDARD 14
  VERSION "${VERSION}"
)

target_include_directories(${TARGET_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})
