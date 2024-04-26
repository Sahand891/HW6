cmake_minimum_required(VERSION 3.26)
project(HW6)

set(CMAKE_CXX_STANDARD 17)

add_executable(HW6 main.cpp
        basis.cpp
        basis.h
        KE.cpp
        KE.h
        KE.h)

include_directories(/Users/sahandadibnia/armadillo-12.6.7/include)

# Find LAPACK
find_package(LAPACK REQUIRED)

# Add Armadillo and LAPACK libraries
find_library(ARMA_LIBRARY armadillo)
target_link_libraries(HW6 ${ARMA_LIBRARY} ${LAPACK_LIBRARIES})
