cmake_minimum_required(VERSION 3.15)
project(plantri C)

set(CMAKE_C_STANDARD 99)

add_executable(plantri src/plantri/plantri.c)

if(UNIX)
    target_link_libraries(plantri m)
endif()

install(TARGETS plantri RUNTIME DESTINATION .)
