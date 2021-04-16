cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_CXX_STANDARD 14)
project(DSFaceBlurSample)

set(NVDS_VERSION 5.0)
set(CUDA_VERSION 10.2)
set(DS_ROOT_DIR /opt/nvidia/deepstream/deepstream-${NVDS_VERSION})
set(DS_LIBS_DIR ${DS_ROOT_DIR}/lib)
set(DS_INCLUDES_DIR ${DS_ROOT_DIR}/sources/includes)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/libs)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/json)

find_package(CUDA REQUIRED)
find_package(GStreamer)
find_package(GLIB2)
find_package(GObject)
find_package(CUDA REQUIRED)
find_package(JsonGlib)
find_package(OpenCV)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

include_directories(${GLIB2_INCLUDE_DIR})
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${GOBJECT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${DS_INCLUDES_DIR})
include_directories(${CMAKE_SOURCE_DIR}/includes/inference)
include_directories(${CMAKE_SOURCE_DIR}/includes/core)
include_directories(${CMAKE_SOURCE_DIR}/includes/utils)
include_directories(${CMAKE_SOURCE_DIR}/external/json/include)
include_directories(${CMAKE_SOURCE_DIR}/external/spdlog/include)
include_directories("${CMAKE_SOURCE_DIR}")

# link directories for app
link_directories(libs)
link_directories(${GLIB_LIBRARY_DIRS})
link_directories(${GSTREAMER_LIBRARY_DIRS})
link_directories(${UUID_LIBRARY_DIRS})
link_directories(${JSON_GLIB_LIBRARY_DIRS})
link_directories(${DS_LIBS_DIR})

file(GLOB INF ${CMAKE_SOURCE_DIR}/src/inference/*.cpp)
file(GLOB CU ${CMAKE_SOURCE_DIR}/src/inference/*.cu)
add_library(yoloinfer SHARED ${INF} ${CU})

add_executable(
        DSFaceBlurSample
        src/core/face_blur_pipeline.cpp
        src/main.cpp
        src/utils/config_parser.cpp
        src/utils/pipelineUtils.cpp        
)

target_link_libraries(yoloinfer nvinfer_plugin nvinfer nvparsers)
target_link_libraries(yoloinfer cudart cublas stdc++fs)

target_link_libraries(DSFaceBlurSample ${GSTREAMER_LIBRARY} ${GLIB2_LIBRARIES} ${GOBJECT_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(DSFaceBlurSample nvdsgst_meta nvds_meta nvbufsurface nvbufsurftransform pthread stdc++fs)
#target_link_libraries(DSFaceBlurSample ${OpenCV_LIBS})
target_link_libraries(DSFaceBlurSample opencv_cudafilters)
