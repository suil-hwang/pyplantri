# pyplantri# pyplantri

plantri를 Python에서 쉽게 사용할 수 있게 해주는 래퍼 패키지입니다.

[plantri](http://users.cecs.anu.edu.au/~bdm/plantri/)는 평면 그래프(triangulation, quadrangulation 등)를 생성하는 고성능 C 프로그램입니다.

## 요구 사항

### Windows 11

다음 중 하나의 C 컴파일러가 필요합니다:

**방법 1: MinGW-w64 (권장)**
```powershell
# winget으로 설치
winget install -e --id MSYS2.MSYS2

# MSYS2 터미널에서
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake

# 또는 standalone MinGW-w64
winget install -e --id niXman.mingw-w64-ucrt
```

PATH에 MinGW bin 폴더를 추가하세요 (예: `C:\msys64\mingw64\bin`)

**방법 2: Visual Studio Build Tools**
```powershell
winget install -e --id Microsoft.VisualStudio.2022.BuildTools
```
설치 시 "C++ 빌드 도구" 워크로드를 선택하세요.

**CMake 설치**
```powershell
winget install -e --id Kitware.CMake
```

## 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/pyplantri.git
cd pyplantri

# 개발 모드로 설치 (plantri 자동 빌드)
pip install -e .
```

설치 과정에서 CMake가 자동으로 plantri를 빌드합니다.

## 핵심 개념

### Q와 Q*의 관계

| 그래프 | 설명 | 정점 수 | 간선 수 |
|--------|------|---------|---------|
| **Q** (Primal) | General Quadrangulation | n+2 | 2n |
| **Q*** (Dual) | 4-regular planar multigraph | n | 2n |

입력값 `n`은 Q*의 정점 수 (구면 위 점의 개수)입니다.

### plantri 옵션

| 옵션 | 설명 | 최소 n |
|------|------|--------|
| `-Q` | General quadrangulation (multigraph) | 3 |
| `-q` | Simple quadrangulation (3-connected) | 8 |
| `-d` | Dual 그래프 출력 |  |
| `-a` | ASCII 형식 출력 |  |
| `-u` | stdout 출력 안 함 (개수만) |  |

## 사용법

### 기본 사용

```python
from pyplantri import Plantri

plantri = Plantri()

# n=5 (Q*: 5정점, Q: 7정점)
# General quadrangulation 생성
output = plantri.run(7, options=["-Q", "-a"])
print(output.decode())

# Dual (4-regular multigraph) 생성
output = plantri.run(7, options=["-Q", "-d", "-a"])
print(output.decode())
```

### example.py 실행

```bash
python example.py
```

n을 입력하면:
- Q* (4-regular multigraph, n정점)
- Q (Quadrangulation, n+2정점)

의 모든 비동형 구조를 출력합니다.

### n별 그래프 개수

| n | plantri 입력 | 비동형 구조 개수 |
|---|--------------|------------------|
| 3 | 5 | 7 |
| 4 | 6 | 30 |
| 5 | 7 | 124 |
| 6 | 8 | 733 |
| 7 | 9 | 4,586 |
| 8 | 10 | 33,373 |

## 프로젝트 구조

```
pyplantri/
├── CMakeLists.txt          # CMake 빌드 설정
├── setup.py                # Python 패키지 설정 (CMake 연동)
├── pyproject.toml          # 프로젝트 메타데이터
├── README.md               # 이 파일
├── example.py              # 사용 예제
├── src/
│   └── plantri/
│       ├── plantri.c       # Plantri 소스 (필수)
│       └── LICENSE-2.0.txt # Apache 2.0 라이선스
└── pyplantri/              # Python 패키지
    ├── __init__.py
    ├── core.py             # 핵심 래퍼 코드
    └── bin/                # 빌드된 실행 파일 (설치 후 생성)
        └── plantri.exe
```

## 문제 해결

### CMake를 찾을 수 없음

```
CMake Error: CMake was unable to find a build program
```

CMake가 PATH에 있는지 확인하세요:
```powershell
cmake --version
```

### 컴파일러를 찾을 수 없음

```
CMake Error: CMAKE_C_COMPILER not set
```

MinGW 또는 Visual Studio Build Tools가 설치되어 있는지 확인하세요:
```powershell
gcc --version  # MinGW
# 또는
cl  # Visual Studio (Developer Command Prompt에서)
```

### plantri 실행 파일을 찾을 수 없음

```python
FileNotFoundError: plantri 실행 파일을 찾을 수 없습니다
```

패키지를 다시 설치하세요:
```bash
pip install -e . --force-reinstall
```

## 라이선스

- pyplantri wrapper: MIT License
- plantri: Apache License 2.0 (원작자: Gunnar Brinkmann, Brendan McKay)

## 참고 자료

- [plantri 공식 페이지](http://users.cecs.anu.edu.au/~bdm/plantri/)
