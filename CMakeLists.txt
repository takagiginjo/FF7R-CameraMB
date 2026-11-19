cmake_minimum_required(VERSION 3.24)
project(ff7_camera_motion_blur LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static CRT so the add-on has no VC++ redist dependency (also used by MinHook below)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(FetchContent)

# ReShade: headers only (no CMakeLists in the repo, SOURCE_SUBDIR stops add_subdirectory)
FetchContent_Declare(reshade
    GIT_REPOSITORY https://github.com/crosire/reshade.git
    GIT_TAG        v6.4.1            # bump if the tag does not exist / to match your ReShade
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES ""
    SOURCE_SUBDIR  _no_cmake)

# NVIDIA NGX headers only (we never link NGX, we read its parameter block)
FetchContent_Declare(dlss
    GIT_REPOSITORY https://github.com/NVIDIA/DLSS.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    GIT_SUBMODULES ""
    SOURCE_SUBDIR  _no_cmake)

FetchContent_Declare(minhook
    GIT_REPOSITORY https://github.com/TsudaKageyu/minhook.git
    GIT_TAG        v1.3.4
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(reshade dlss minhook)

add_library(ff7_camera_motion_blur SHARED src/addon.cpp)

target_compile_definitions(ff7_camera_motion_blur PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(ff7_camera_motion_blur PRIVATE
    ${reshade_SOURCE_DIR}/include
    ${dlss_SOURCE_DIR}/include)
target_link_libraries(ff7_camera_motion_blur PRIVATE minhook d3dcompiler)

set_target_properties(ff7_camera_motion_blur PROPERTIES
    PREFIX ""
    SUFFIX ".addon64")
