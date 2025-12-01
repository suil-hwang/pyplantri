# pyplantri

plantri를 Python에서 쉽게 사용할 수 있게 해주는 래퍼 패키지입니다.

[plantri](http://users.cecs.anu.edu.au/~bdm/plantri/)는 평면 그래프(triangulation, quadrangulation 등)를 생성하는 C 프로그램입니다.

## 논문 기반 사용 (Samuel Peltier et al., 2021)

이 패키지는 **Simple Quadrangulation on a Sphere (SQS)**와 그 **Dual Graph**를 생성하기 위해 설계되었습니다.

### Dual Q*의 성질

| # | 성질 | 설명 |
|---|------|------|
| 1 | Planar graph | 평면 그래프 |
| 2 | Multigraph | Double edge 허용 |
| 3 | Loop 없음 | Q에 degree 1 vertex가 없으므로 |
| 4 | 4-regular | 모든 vertex가 정확히 degree 4 |

### Q와 Q*의 관계 (Euler 공식)

| 그래프 | 설명 | 정점 수 |
|--------|------|---------|
| **Q*** (Dual) | 4-regular planar multigraph (loop-free) | n |
| **Q** (Primal) | Simple Quadrangulation | n + 2 |

**입력 규칙:** `example.py`와 `pyplantri_ilp_bridge.py`의 입력 `n`은 **Q* (Dual)의 정점 수**입니다. 내부적으로 plantri에는 `n + 2` (Q의 정점 수)가 전달됩니다.


## 설치

```bash
git clone https://github.com/suil-hwang/pyplantri.git
cd pyplantri

pip install -e .
```

설치 과정에서 CMake가 자동으로 plantri를 빌드합니다.

## 사용법

### 기본 사용

```python
from pyplantri import Plantri

plantri = Plantri()

# Simple Quadrangulation 생성 (6정점, -q -c2 -m2 옵션)
output = plantri.run(6, options=["-q", "-c2", "-m2", "-a"])
print(output.decode())

# Dual Graph 생성 (4-regular planar multigraph, loop-free)
output = plantri.run(6, options=["-q", "-c2", "-m2", "-d", "-a"])
print(output.decode())
```

### example.py 

```bash
# n = Q*의 정점 수 (최소 3)
python example.py 3    # Q*: 3정점, Q: 5정점

# 내부적으로 plantri 호출:
# plantri -q -c2 -m2 -a 5    → Q 출력 (5정점)
# plantri -q -c2 -m2 -a -d 5 → Q* 출력 (3정점)
```

### plantri 옵션

| 옵션 | 설명 | 비고 |
|------|------|------|
| `-q` | Simple quadrangulation | **권장** (논문 조건 충족) |
| `-Q` | General quadrangulation | Dual에 loop 발생 가능 |
| `-d` | Dual 그래프 출력 | |
| `-a` | ASCII 형식 출력 | |
| `-c2` | 2-connected | `-q`와 함께 사용 |
| `-m2` | minimum degree 2 | `-q`와 함께 사용 |

**논문 조건 만족을 위한 옵션: `-q -c2 -m2`**
- `-q`: Simple quadrangulation (primal에 multi-edge가 없도록)
- `-c2`: 2-connected
- `-m2`: minimum degree 2
- → Dual: 4-edge-connected quartic multigraph (double edge 허용, loop 없음)

### n별 그래프 개수 (-q -c2 -m2)

| n (Q* 정점 수) | Q 정점 수 | 비동형 구조 개수 |
|----------------|-----------|------------------|
| 2 | 4 | 1 |
| 3 | 5 | 1 |
| 4 | 6 | 2 |
| 6 | 8 | 9 |
| 8 | 10 | 62 |
| 10 | 12 | 803 |
| 12 | 14 | 15,882 |
| 14 | 16 | 393,075 |
| 16 | 18 | 10,938,182 |
| 18 | 20 | 326,258,544 |

## 프로젝트 구조

```
pyplantri/
├── CMakeLists.txt          # CMake 빌드 설정
├── setup.py                # Python 패키지 설정 (CMake 연동)
├── pyproject.toml          # 프로젝트 메타데이터
├── README.md               # 이 파일
├── examples/
│   └──example.py              # Q/Q* 생성 예제
├── src/
│   └── plantri/
│       ├── plantri.c       # Plantri 소스 (필수)
│       └── LICENSE-2.0.txt # Apache 2.0 라이선스
└── pyplantri/              # Python 패키지
    ├── __init__.py
    ├── core.py             # Plantri 래퍼 및 GraphConverter 클래스
    └── bin/                # 빌드된 실행 파일 (설치 후 생성)
        └── plantri.exe
```

## 라이선스

- pyplantri wrapper: MIT License
- plantri: Apache License 2.0 (원작자: Gunnar Brinkmann, Brendan McKay)

## Reference

- [plantri 공식 페이지](http://users.cecs.anu.edu.au/~bdm/plantri/)
- Samuel Peltier et al. (2021) - Tubular parametric volume objects: Thickening a piecewise smooth 3D stick figure
