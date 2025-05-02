cmake_minimum_required(VERSION 3.20)

project(akari_cpp_ext LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(tiny-cuda-nn)
#TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT
set(CMAKE_CUDA_ARCHITECTURES ${TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT})
file(READ "../../../blender_src_path.txt" BLENDER_SRC_PATH)
string(STRIP ${BLENDER_SRC_PATH} BLENDER_SRC_PATH)
if(WIN32)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "/arch:AVX2 /WX  ${CMAKE_CXX_FLAGS}")
    endif()
else()
    set(CMAKE_CXX_FLAGS "-lstdc++ -pthread -mavx2 ${CMAKE_CXX_FLAGS}")
endif()

add_library(akari_cpp_ext SHARED akari_cpp_ext.cpp rgb2spec_opt.cpp  akari_tcnn.cu)
target_link_libraries(akari_cpp_ext PUBLIC tiny-cuda-nn)
target_include_directories(akari_cpp_ext PUBLIC
    ${BLENDER_SRC_PATH}/source/blender/
    ${BLENDER_SRC_PATH}/source/blender/makesdna/
    ${BLENDER_SRC_PATH}/source/blender/blenlib/
    ${BLENDER_SRC_PATH}/source/blender/blenkernel/
    ${BLENDER_SRC_PATH}/intern/
    ${BLENDER_SRC_PATH}/intern/guardedalloc
)