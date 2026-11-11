# Thalia - CMake build
cmake_minimum_required(VERSION 3.10)
project(thalia CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O2 -Wall -Wextra)

find_package(OpenSSL REQUIRED)

add_executable(thalia
    src/hdc.cpp
    src/tokenize.cpp
    src/encoder.cpp
    src/self.cpp
    src/memory.cpp
    src/semantics.cpp
    src/inference.cpp
    src/engine.cpp src/speaker.cpp
    src/main.cpp
)

target_include_directories(thalia PRIVATE include)
target_link_libraries(thalia PRIVATE OpenSSL::Crypto)

install(TARGETS thalia RUNTIME DESTINATION bin)
