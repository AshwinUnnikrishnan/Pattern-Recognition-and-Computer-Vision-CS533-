cmake_minimum_required(VERSION 2.8...3.13)
project(retrieval)
find_package(OpenCV REQUIRED)

set( NAME_SRC
    src/retrieval.cpp    
)
set( NAME_HEADERS
     include/cvui.h
     include/fileLib.h
     include/filter.h

)
set( NAME_LIBRARY
     lib/fileLib.cpp
     lib/filter.cpp
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(wxWidgets-3.1.5)

include_directories(${OpenCV_INCLUDE_DIRS} "include/" )
add_executable( retrieval ${NAME_SRC} ${NAME_HEADERS} ${NAME_LIBRARY})

target_link_libraries(retrieval ${OpenCV_LIBS} wx::net wx::core wx::base)
