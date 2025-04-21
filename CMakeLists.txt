cmake_minimum_required(VERSION 3.11)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
project(TensorRT_MODNet LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
include_directories(/usr/local/TensorRT-8.5.1.7/include)

set(CMAKE_PREFIX_PATH /root/opencv_4.5.5)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda-11.8/include)
include_directories(${TensorRT_SAMPLE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/src)

link_directories(/usr/local/cuda-11.8/lib64/)

add_library(engine SHARED
    src/engine.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
)
target_link_libraries(engine PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(engine PROPERTIES CUDA_ARCHITECTURES "61")

add_library(utils SHARED
    src/utils/preprocess.cpp
    src/utils/draw.cpp
)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})
set_target_properties(utils PROPERTIES CUDA_ARCHITECTURES "61")


add_executable(build
    build.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build
    PRIVATE
    -Wl,--no-as-needed,--gc-sections,--unresolved-symbols=ignore-in-shared-libs
    gflags
    #utils
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    )
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61")


add_library(algo SHARED
    src/modnet.cpp
)
target_link_libraries(algo engine utils cuda cudart opencv_dnn)


add_executable(modnet_test
    modnet_test.cpp
)
target_link_libraries(modnet_test utils algo gflags )