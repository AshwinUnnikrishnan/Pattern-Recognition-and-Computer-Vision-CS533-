cmake_minimum_required(VERSION 2.8)
project(startProject)
find_package(OpenCV REQUIRED)

set( NAME_SRC
    src/vidDisplay.cpp    
)
set( NAME_HEADERS
     include/filter.h
)
set( NAME_LIBRARY
     lib/filter.cpp
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories((${OpenCV_INCLUDE_DIRS}) "include/")
link_directories( ${CMAKE_BINARY_DIR}/bin)
add_executable( startProject ${NAME_SRC} ${NAME_HEADERS} ${NAME_LIBRARY})

target_link_libraries(startProject ${OpenCV_LIBS})
