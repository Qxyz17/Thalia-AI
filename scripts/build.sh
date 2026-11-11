#!/usr/bin/env bash
# Build Thalia directly with g++, no CMake needed.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
SRCS="src/hdc.cpp src/tokenize.cpp src/encoder.cpp src/self.cpp src/memory.cpp \
src/semantics.cpp src/inference.cpp src/engine.cpp src/speaker.cpp src/main.cpp"
# On Windows a dynamic build needs libcrypto/libgcc/libstdc++ DLLs next to
# the exe; link statically so build/thalia runs on a bare system.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        g++ -std=c++17 -O2 -Wall -Wextra -Iinclude -static $SRCS \
            -l:libcrypto.a -lws2_32 -lcrypt32 -lbcrypt -o build/thalia.exe
        echo "Built: build/thalia.exe" ;;
    *)
        g++ -std=c++17 -O2 -Wall -Wextra -Iinclude $SRCS \
            -lcrypto -o build/thalia
        echo "Built: build/thalia" ;;
esac
