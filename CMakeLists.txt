cmake_minimum_required(VERSION 3.15)
project(plantri C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(plantri_sqs src/plantri_sqs.c)

if(UNIX)
    target_link_libraries(plantri_sqs m)
endif()

install(TARGETS plantri_sqs RUNTIME DESTINATION .)
