cmake_minimum_required(VERSION 3.20)
project(ff7blur LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# fxc comes with the Windows SDK. Outside a developer prompt, take the newest
# SDK installed in the default place.
file(GLOB SDK_BIN LIST_DIRECTORIES true "C:/Program Files (x86)/Windows Kits/10/bin/10.*/x64")
list(SORT SDK_BIN)
list(REVERSE SDK_BIN)
find_program(FXC fxc REQUIRED HINTS "$ENV{WindowsSdkVerBinPath}x64" ${SDK_BIN})

set(header ${CMAKE_CURRENT_BINARY_DIR}/shaders/flatten.h)
add_custom_command(
    OUTPUT ${header}
    COMMAND ${FXC} /nologo /T cs_5_0 /E main /O3 /Vn flatten_cs /Fh ${header} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/flatten.hlsl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/flatten.hlsl
    COMMENT "Compiling flatten.hlsl")

# The leading 00- makes ReShade load the add-on first.
add_library(ff7blur SHARED src/ff7blur.cpp ${header})
target_include_directories(ff7blur PRIVATE
    deps/reshade/include
    deps/imgui
    ${CMAKE_CURRENT_BINARY_DIR}/shaders)
target_link_libraries(ff7blur PRIVATE d3d11 dxgi)
target_compile_definitions(ff7blur PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
target_compile_options(ff7blur PRIVATE /W3 /utf-8)
set_target_properties(ff7blur PROPERTIES OUTPUT_NAME "00-FF7Remake-CameraBlur" SUFFIX ".addon64")
