cmake_minimum_required(VERSION 3.15)
project(plantri C)

set(CMAKE_C_STANDARD 99)

# plantri 실행 파일 빌드
add_executable(plantri src/plantri/plantri.c)

# Unix에서 math 라이브러리 링크
if(UNIX)
    target_link_libraries(plantri m)
endif()

# 설치 경로 설정
install(TARGETS plantri
        RUNTIME DESTINATION .)
