cmake_minimum_required(VERSION 3.10)

PROJECT(ASimpleNNCMake)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

INCLUDE_DIRECTORIES(include)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

set(mySRC_DIR ${DIR_SRCS})

ADD_EXECUTABLE(${PROJECT_NAME} ${mySRC_DIR})



