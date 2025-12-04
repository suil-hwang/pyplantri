cmake_minimum_required(VERSION 3.15)
project(plantri C)

set(CMAKE_C_STANDARD 99)

# plantri 실행 파일 빌드
add_executable(plantri src/plantri/plantri.c)

# Unix에서 math 라이브러리 링크
if(UNIX)
    target_link_libraries(plantri m)
endif()

# 실행 파일을 bin 폴더에 설치 (scikit-build-core가 wheel.install-dir과 결합)
install(TARGETS plantri RUNTIME DESTINATION .)
