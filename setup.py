import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


class CMakeBuild:
    """CMake 빌드 헬퍼"""

    @staticmethod
    def build(source_dir: Path, install_dir: Path):
        build_dir = source_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # CMake 설정
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # Windows용 설정
        if sys.platform == "win32":
            # Visual Studio 또는 Ninja 사용
            # MinGW가 있으면 MinGW 사용
            if shutil.which("gcc"):
                cmake_args.extend(["-G", "MinGW Makefiles"])
            else:
                # Visual Studio 버전 자동 감지
                cmake_args.extend(["-A", "x64"])

        # Configure
        subprocess.check_call(
            ["cmake", str(source_dir)] + cmake_args,
            cwd=build_dir
        )

        # Build
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"],
            cwd=build_dir
        )

        # Install
        subprocess.check_call(
            ["cmake", "--install", ".", "--config", "Release"],
            cwd=build_dir
        )


def _build_plantri():
    """plantri 빌드 실행"""
    source_dir = Path(__file__).parent.absolute()
    install_dir = source_dir / "pyplantri" / "bin"
    install_dir.mkdir(parents=True, exist_ok=True)

    CMakeBuild.build(source_dir, install_dir)


class BuildPyWithCMake(build_py):
    def run(self):
        _build_plantri()
        super().run()


class DevelopWithCMake(develop):
    def run(self):
        _build_plantri()
        super().run()


class InstallWithCMake(install):
    def run(self):
        _build_plantri()
        super().run()


setup(
    name="pyplantri",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pyplantri": ["bin/*", "bin/**/*"],
    },
    cmdclass={
        "build_py": BuildPyWithCMake,
        "develop": DevelopWithCMake,
        "install": InstallWithCMake,
    },
)
