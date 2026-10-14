# gq_spz: native PLY -> SPZ converter for GaussQueue, built on Niantic's spz library.
# Native code has no WebAssembly 4 GB memory ceiling, so it handles 10M+ splat captures
# that the WebAssembly encoder (used by SplatTransform) aborts on.
#
#   cmake -S tools/spz-native -B build/spz-native -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/spz-native --config Release
cmake_minimum_required(VERSION 3.16)
project(gq_spz CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(spz
  GIT_REPOSITORY https://github.com/nianticlabs/spz.git
  GIT_TAG        5bf2945de1a003cee07133b1e495fe9c6ffdc7e7   # v3.0.0
  GIT_SHALLOW    FALSE)
set(SPZ_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(SPZ_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(SPZ_BUILD_EXTENSIONS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spz)

add_executable(gq_spz gq_spz.cpp)
target_link_libraries(gq_spz PRIVATE spz)
target_include_directories(gq_spz PRIVATE ${spz_SOURCE_DIR}/src)
