# src/pyplantri/plantri.py
from __future__ import annotations

import os
import re
import subprocess
import sysconfig
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import which
from collections.abc import Iterator
from typing import Literal, NoReturn

from .converter import GraphConverter
from .types import EdgeLabel, EdgeLabelPairs, HalfEdge


def _summarize_process_text(text: str, *, limit: int = 400) -> str:
    """Collapse process output into a short single-line excerpt."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


@dataclass(frozen=True, slots=True)
class ParsedGraphSection:
    """One graph section parsed from plantri -T double_code output."""

    vertex_count: int
    cyclic_adjacency: dict[int, list[int]]
    twin_map: dict[HalfEdge, HalfEdge]
    edge_label_pairs: EdgeLabelPairs


_LINE_ORIENTED_OUTPUT_FLAGS = frozenset("agsT")
_OUTPUT_FLAGS = frozenset("agsETu")
_DOUBLE_CODE_OUTPUT_FLAG = frozenset("T")
_QUADRANGULATION_FLAGS = frozenset("qQ")
# Sorted because frozenset iteration order varies per process; the error message must not.
_LINE_ORIENTED_FLAG_HINT = "/".join(f"-{flag}" for flag in sorted(_LINE_ORIENTED_OUTPUT_FLAGS))


def _has_plantri_flag(options: list[str] | None, flags: frozenset[str]) -> bool:
    """Return whether separate or combined plantri options contain a flag."""
    return any(
        option.startswith("-") and any(char in flags for char in option[1:])
        for option in (options or [])
    )


def _selected_plantri_flags(
    options: list[str] | None,
    flags: frozenset[str],
) -> set[str]:
    """Return selected boolean plantri flags, including combined options."""
    return {
        char
        for option in (options or [])
        if option.startswith("-")
        for char in option[1:]
        if char in flags
    }


def _without_plantri_flags(
    options: list[str] | None,
    flags: frozenset[str],
) -> list[str]:
    """Remove boolean plantri flags while preserving combined option tokens."""
    normalized: list[str] = []
    for option in options or []:
        if not option.startswith("-") or option == "-":
            normalized.append(option)
            continue
        remaining = "".join(char for char in option[1:] if char not in flags)
        if remaining:
            normalized.append(f"-{remaining}")
    return normalized


def _validate_n_vertices(n_vertices: int) -> None:
    """Require a positive integer vertex count for public plantri commands."""
    if type(n_vertices) is not int or n_vertices <= 0:
        raise ValueError(f"plantri: n_vertices must be a positive int; got {n_vertices!r}")


def _build_tag_sort_key(tag_dir: Path) -> tuple[bool, str]:
    """Prefer build directories for the current platform, then sort by name."""
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return not tag_dir.name.endswith(platform_tag), tag_dir.name


def _is_executable(path: Path) -> bool:
    """Return whether path names a runnable executable on this platform."""
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _find_plantri_exe() -> Path:
    """Finds the plantri executable path."""
    exe_name = "plantri.exe" if os.name == "nt" else "plantri"

    # Package bin folder (installed).
    pkg_bin = Path(__file__).parent / "bin" / exe_name
    if _is_executable(pkg_bin):
        return pkg_bin

    # scikit-build-core build folder (editable/dev): plantri.py -> pyplantri -> src -> project_root.
    project_root = Path(__file__).parent.parent.parent
    build_dir = project_root / "build"
    if build_dir.is_dir():
        for tag_dir in sorted(build_dir.iterdir(), key=_build_tag_sort_key):
            if tag_dir.is_dir():
                # Release folder (Visual Studio build).
                release_exe = tag_dir / "Release" / exe_name
                if _is_executable(release_exe):
                    return release_exe
                # MinGW/Unix build.
                direct_exe = tag_dir / exe_name
                if _is_executable(direct_exe):
                    return direct_exe

    path_exe = which(exe_name)
    if path_exe is not None:
        resolved_path_exe = Path(path_exe)
        if _is_executable(resolved_path_exe):
            return resolved_path_exe

    # Default path for error messages.
    return Path(__file__).parent / "bin" / exe_name


# Bundled plantri limits. Count mode (-u) uses the full MAXN=64 range.
# The fixed -T output buffer is safe through N=57 for quadrangulations
# (E=2N-4), or N=39 for the densest built-in planar class (E<=3N-6).
_BUNDLED_PLANTRI_MAX_N = 64
_QUADRANGULATION_DOUBLE_CODE_MAX_N = 57
_GENERAL_PLANAR_DOUBLE_CODE_MAX_N = 39

# Public SQS dual bounds; n=N-2 for a quadrangulation with N primal vertices.
MIN_DUAL_VERTEX_COUNT = 3
MAX_DUAL_VERTEX_COUNT = _BUNDLED_PLANTRI_MAX_N - 2
MAX_DOUBLE_CODE_DUAL_VERTEX_COUNT = (_QUADRANGULATION_DOUBLE_CODE_MAX_N - 2)


class PlantriError(Exception):
    """Plantri execution failure."""


class PlantriExecutableNotFoundError(PlantriError, FileNotFoundError):
    """Plantri executable could not be found."""


class QuadrangulationDualClass(str, Enum):
    """Dual graph classes available from plantri quadrangulation modes."""

    QUARTIC_MULTIGRAPH = "quartic_multigraph"
    SIMPLE_QUARTIC = "simple_quartic"


def _raise_executable_not_found(executable: Path) -> NoReturn:
    raise PlantriExecutableNotFoundError(f"plantri: executable not found {executable}; run 'pip install -e .' or add plantri to PATH")


def _raise_executable_not_runnable(
    executable: Path,
    error: OSError | None = None,
) -> NoReturn:
    message = f"plantri: executable is not runnable {executable}"
    if error is not None:
        message = f"{message}: {error}"
    raise PlantriError(message) from error


class Plantri:
    """Wrapper for the plantri executable."""
    _COUNT_INCOMPATIBLE_OUTPUT_FLAGS = _OUTPUT_FLAGS

    def __init__(self, executable: Path | None = None) -> None:
        """Initializes Plantri with the executable path."""
        self.executable = (
            Path(executable).expanduser().resolve()
            if executable is not None
            else _find_plantri_exe()
        )
        if not self.executable.is_file():
            _raise_executable_not_found(self.executable)
        if not _is_executable(self.executable):
            _raise_executable_not_runnable(self.executable)

    def run(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> bytes:
        """Runs plantri with the given parameters."""
        cmd = self._build_command(n_vertices, options=options, output_format=output_format)

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors="replace") if e.stderr else str(e)
            raise PlantriError(f"plantri: execution failed (exit {e.returncode}); {_summarize_process_text(stderr_text)}") from e
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
        except OSError as e:
            _raise_executable_not_runnable(self.executable, e)

    def _build_command(
        self,
        n_vertices: int,
        *,
        options: list[str] | None,
        output_format: Literal["planar_code", "ascii"],
    ) -> list[str]:
        """Builds a plantri command line for the given options."""
        if output_format not in ("planar_code", "ascii"):
            raise ValueError(f"plantri: unsupported output_format {output_format!r}; use 'planar_code' or 'ascii'")
        _validate_n_vertices(n_vertices)
        selected_output_flags = _selected_plantri_flags(options, _OUTPUT_FLAGS)
        if output_format == "ascii":
            selected_output_flags.add("a")
        if len(selected_output_flags) > 1:
            formatted_flags = ", ".join(
                f"-{flag}" for flag in sorted(selected_output_flags)
            )
            raise ValueError(f"plantri: conflicting output flags: {formatted_flags}")
        self._validate_double_code_vertex_count(n_vertices, options)

        cmd = [str(self.executable)]
        if options:
            cmd.extend(options)

        # Set output format flag.
        if output_format == "ascii" and not _has_plantri_flag(options, frozenset("a")):
            cmd.append("-a")

        cmd.append(str(n_vertices))
        return cmd

    @staticmethod
    def _validate_double_code_vertex_count(
        n_vertices: int,
        options: list[str] | None,
    ) -> None:
        """Reject sizes that can overflow the bundled executable's -T buffer."""
        if not _has_plantri_flag(options, _DOUBLE_CODE_OUTPUT_FLAG):
            return

        is_quadrangulation = _has_plantri_flag(
            options,
            _QUADRANGULATION_FLAGS,
        )
        maximum = (
            _QUADRANGULATION_DOUBLE_CODE_MAX_N
            if is_quadrangulation
            else _GENERAL_PLANAR_DOUBLE_CODE_MAX_N
        )
        if n_vertices > maximum:
            graph_class = "quadrangulation" if is_quadrangulation else "planar"
            raise ValueError(
                "plantri: n_vertices unsupported for "
                f"{graph_class} double_code generation: "
                f"{n_vertices} > {maximum}"
            )

    def iter_stdout_lines(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> Iterator[bytes]:
        """Stream non-empty stdout lines for line-oriented plantri output."""
        cmd = self._build_command(n_vertices, options=options, output_format=output_format)
        # Binary planar_code may contain newline bytes, so streaming requires a line format.
        if output_format != "ascii" and not _has_plantri_flag(options, _LINE_ORIENTED_OUTPUT_FLAGS):
            raise ValueError(f"plantri: iter_stdout_lines requires line-oriented output; use output_format='ascii' or a {_LINE_ORIENTED_FLAG_HINT} flag")

        # Spool stderr to disk so long streams cannot deadlock on a full stderr pipe.
        with tempfile.TemporaryFile() as stderr_file:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                )
            except FileNotFoundError as e:
                raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
            except OSError as e:
                _raise_executable_not_runnable(self.executable, e)

            if proc.stdout is None:
                proc.kill()
                proc.wait()
                raise PlantriError("plantri: failed to capture stdout")

            fully_consumed = False
            try:
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if line:
                        yield line
                fully_consumed = True
            finally:
                proc.stdout.close()

                # Only natural EOF makes nonzero exit authoritative; early close terminates plantri.
                if fully_consumed:
                    return_code = proc.wait()
                    if return_code != 0:
                        stderr_file.seek(0)
                        stderr_text = stderr_file.read().decode(
                            "utf-8",
                            errors="replace",
                        )
                        stderr_excerpt = _summarize_process_text(
                            stderr_text,
                            limit=4000,
                        )
                        raise PlantriError(
                            "plantri: execution failed "
                            f"(exit {return_code}); {stderr_excerpt}"
                        )
                else:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()

    def count_from_options(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        timeout: float = 3600.0,
    ) -> int:
        """Counts graphs with arbitrary generation options via plantri ``-u``."""
        _validate_n_vertices(n_vertices)
        normalized_options = _without_plantri_flags(
            options,
            self._COUNT_INCOMPATIBLE_OUTPUT_FLAGS,
        )

        try:
            result = subprocess.run(
                [str(self.executable)] + normalized_options + ["-u", str(n_vertices)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise PlantriError(f"plantri: timed out after {timeout}s for n={n_vertices}, options={normalized_options}") from e
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
        except OSError as e:
            _raise_executable_not_runnable(self.executable, e)

        if result.returncode != 0:
            stderr_excerpt = _summarize_process_text(result.stderr)
            stdout_excerpt = _summarize_process_text(result.stdout)
            raise PlantriError(f"plantri: count failed (exit {result.returncode}); stderr={stderr_excerpt}; stdout={stdout_excerpt}")

        # The output class name varies by mode; only the trailing verb is stable.
        for line in reversed(result.stderr.splitlines()):
            match = re.match(
                r"^\s*(\d+)\s+.*\b(?:generated|written)\b",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                return int(match.group(1))

        stderr_excerpt = _summarize_process_text(result.stderr)
        stdout_excerpt = _summarize_process_text(result.stdout)
        raise PlantriError(f"plantri: count parse failed; stderr={stderr_excerpt}; stdout={stdout_excerpt}")


class QuadrangulationEnumerator:
    """Enumerates dual quartic plane multigraphs of simple quadrangulations.

    Uses plantri quadrangulation modes in double_code format.

    - `QUARTIC_MULTIGRAPH`: `-q -c2 -m2 -T`
    - `SIMPLE_QUARTIC`: `-q -c2 -T`
    """

    _FLAGS_BY_DUAL_CLASS: dict[QuadrangulationDualClass, list[str]] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: ["-q", "-c2", "-m2", "-T"],
        QuadrangulationDualClass.SIMPLE_QUARTIC: ["-q", "-c2", "-T"],
    }
    _MIN_NONEMPTY_DUAL_VERTICES: dict[QuadrangulationDualClass, int] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: MIN_DUAL_VERTEX_COUNT,
        QuadrangulationDualClass.SIMPLE_QUARTIC: 6,
    }

    def __init__(self) -> None:
        """Initialize the SQS enumerator."""
        # Delay executable resolution so known-empty requests do not require plantri.
        self._plantri: Plantri | None = None

    def _get_plantri(self) -> Plantri:
        if self._plantri is None:
            self._plantri = Plantri()
        return self._plantri

    @staticmethod
    def _normalize_dual_class(
        dual_class: QuadrangulationDualClass | str,
    ) -> QuadrangulationDualClass:
        if isinstance(dual_class, QuadrangulationDualClass):
            return dual_class
        try:
            return QuadrangulationDualClass(dual_class)
        except ValueError as exc:
            raise ValueError(f"unsupported quadrangulation dual_class: {dual_class!r}") from exc

    @classmethod
    def _flags_for_dual_class(
        cls,
        dual_class: QuadrangulationDualClass,
    ) -> list[str]:
        return list(cls._FLAGS_BY_DUAL_CLASS[dual_class])

    @staticmethod
    def _dual_class_from_filter(
        *,
        double_edge_free_only: bool,
    ) -> QuadrangulationDualClass:
        if double_edge_free_only:
            return QuadrangulationDualClass.SIMPLE_QUARTIC
        return QuadrangulationDualClass.QUARTIC_MULTIGRAPH

    @classmethod
    def _min_nonempty_dual_vertices(
        cls,
        dual_class: QuadrangulationDualClass,
    ) -> int:
        return cls._MIN_NONEMPTY_DUAL_VERTICES[dual_class]

    @staticmethod
    def _validate_supported_dual_vertex_count(dual_vertex_count: int) -> None:
        """Reject dual sizes outside the bundled plantri count range."""
        if type(dual_vertex_count) is not int:
            raise ValueError(f"dual_vertex_count must be an integer; got {dual_vertex_count!r}")
        if dual_vertex_count < MIN_DUAL_VERTEX_COUNT:
            raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count} < {MIN_DUAL_VERTEX_COUNT}")
        if dual_vertex_count > MAX_DUAL_VERTEX_COUNT:
            raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count} > {MAX_DUAL_VERTEX_COUNT} (bundled plantri MAXN={_BUNDLED_PLANTRI_MAX_N})")

    @classmethod
    def _validate_supported_double_code_dual_vertex_count(
        cls,
        dual_vertex_count: int,
    ) -> None:
        """Reject dual sizes unsafe for bundled quadrangulation double_code."""
        cls._validate_supported_dual_vertex_count(dual_vertex_count)
        if dual_vertex_count > MAX_DOUBLE_CODE_DUAL_VERTEX_COUNT:
            raise ValueError(f"dual_vertex_count unsupported for double_code generation: {dual_vertex_count} > {MAX_DOUBLE_CODE_DUAL_VERTEX_COUNT}")

    def generate_pairs(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> Iterator[tuple[ParsedGraphSection, ParsedGraphSection]]:
        """Yield (primal, dual) pairs from plantri."""
        for line in self.iter_double_code_lines(dual_vertex_count, dual_class=dual_class):
            yield self.parse_double_code(line)

    def count(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> int:
        """Count plane-map isomorphism classes, identifying mirror images."""
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = self._normalize_dual_class(dual_class)
        if dual_vertex_count < self._min_nonempty_dual_vertices(resolved_dual_class):
            return 0
        primal_vertex_count = dual_vertex_count + 2
        return self._get_plantri().count_from_options(
            primal_vertex_count,
            options=self._flags_for_dual_class(resolved_dual_class),
        )

    def iter_double_code_lines(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> Iterator[bytes]:
        """Yield raw double_code lines as bytes from plantri stdout."""
        self._validate_supported_double_code_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = self._normalize_dual_class(dual_class)
        if dual_vertex_count < self._min_nonempty_dual_vertices(resolved_dual_class):
            return
        # Euler's formula for plane graphs: V - E + F = 2
        # For quadrangulations: primal_vertices = dual_vertices + 2
        primal_vertex_count = dual_vertex_count + 2
        yield from self._get_plantri().iter_stdout_lines(
            primal_vertex_count,
            self._flags_for_dual_class(resolved_dual_class),
        )

    @staticmethod
    def parse_double_code(
        double_code_line: str | bytes,
    ) -> tuple[ParsedGraphSection, ParsedGraphSection]:
        """Parse a plantri double_code line into (primal, dual) sections.

        Without -d, plantri outputs primal first then dual. With -d
        the order is reversed. This method detects the section order via
        4-regularity and vertex-count checks. It expects authentic plantri
        -T output rather than validating planar duality independently.
        """
        raw = (
            double_code_line.encode("latin-1")
            if isinstance(double_code_line, str)
            else double_code_line
        )
        parts = list(raw.split())
        first_vertex_count, first_edge_lists, next_idx = QuadrangulationEnumerator._parse_section(parts, 0, "first")
        second_vertex_count, second_edge_lists, next_idx = QuadrangulationEnumerator._parse_section(parts, next_idx, "second")
        if next_idx != len(parts):
            raise ValueError(f"double_code trailing token count: {len(parts) - next_idx}")

        first_data = QuadrangulationEnumerator._build_section(first_vertex_count, first_edge_lists)
        second_data = QuadrangulationEnumerator._build_section(second_vertex_count, second_edge_lists)
        return QuadrangulationEnumerator._resolve_primal_dual_sections(
            first_data,
            second_data,
        )

    @staticmethod
    def _parse_section(
        parts: list[bytes],
        start_idx: int,
        section_name: str,
    ) -> tuple[int, list[bytes], int]:
        """Parse one double_code section header and its edge-label tokens."""
        if start_idx >= len(parts):
            raise ValueError(f"double_code missing {section_name} section header")

        vertex_count = int(parts[start_idx])
        if vertex_count < 0:
            raise ValueError(f"double_code {section_name} count invalid: {vertex_count}")
        idx = start_idx + 1
        end_idx = idx + vertex_count
        if end_idx > len(parts):
            raise ValueError(f"double_code {section_name} count mismatch: {len(parts) - idx} != {vertex_count}")
        edge_lists = parts[idx:end_idx]

        return vertex_count, edge_lists, end_idx

    @staticmethod
    def _build_section(
        vertex_count: int,
        edge_lists: list[bytes],
    ) -> ParsedGraphSection:
        """Build one parsed section from edge-label token lists."""
        adjacency, twin_map, edge_label_pairs = QuadrangulationEnumerator._build_adjacency_and_twins(edge_lists)
        return ParsedGraphSection(
            vertex_count=vertex_count,
            cyclic_adjacency=adjacency,
            twin_map=twin_map,
            edge_label_pairs=edge_label_pairs,
        )

    @staticmethod
    def _resolve_primal_dual_sections(
        first_data: ParsedGraphSection,
        second_data: ParsedGraphSection,
    ) -> tuple[ParsedGraphSection, ParsedGraphSection]:
        """Classify the two sections as `(primal, dual)`."""
        QuadrangulationEnumerator._validate_cross_section_edge_labels(first_data, second_data)

        first_is_4_regular = GraphConverter.is_4_regular(first_data.cyclic_adjacency)
        second_is_4_regular = GraphConverter.is_4_regular(second_data.cyclic_adjacency)

        if first_is_4_regular == second_is_4_regular:
            raise ValueError(f"double_code quartic classification invalid: ({first_is_4_regular}, {second_is_4_regular})")

        if first_is_4_regular:
            dual_data, primal_data = first_data, second_data
        else:
            dual_data, primal_data = second_data, first_data

        if primal_data.vertex_count != dual_data.vertex_count + 2:
            raise ValueError(f"double_code primal/dual vertex mismatch: primal={primal_data.vertex_count}, dual={dual_data.vertex_count}")

        return primal_data, dual_data

    @staticmethod
    def _validate_cross_section_edge_labels(
        first_data: ParsedGraphSection,
        second_data: ParsedGraphSection,
    ) -> None:
        """Check that both sections describe the same labeled edge set."""
        first_labels = set(first_data.edge_label_pairs)
        second_labels = set(second_data.edge_label_pairs)
        if first_labels != second_labels:
            missing_in_second = sorted(
                QuadrangulationEnumerator._format_edge_name_for_error(label)
                for label in first_labels - second_labels
            )
            missing_in_first = sorted(
                QuadrangulationEnumerator._format_edge_name_for_error(label)
                for label in second_labels - first_labels
            )
            raise ValueError(f"double_code edge label mismatch: first-only={missing_in_second}, second-only={missing_in_first}")

    @staticmethod
    def _format_edge_name_for_error(edge_name: str | int) -> str:
        """Formats an edge label for stable, readable error messages."""
        if isinstance(edge_name, str):
            return edge_name
        if 32 <= edge_name <= 126:
            return chr(edge_name)
        return f"0x{edge_name:02x}"

    @staticmethod
    def _build_adjacency_and_twins(
        edge_lists: list[bytes],
    ) -> tuple[
        dict[int, list[int]],
        dict[HalfEdge, HalfEdge],
        EdgeLabelPairs,
    ]:
        """Build adjacency, twin map, and edge-label/half-edge pairs."""
        occurrences: dict[EdgeLabel, list[HalfEdge]] = {}
        for vertex, labels in enumerate(edge_lists, start=1):
            for slot, label in enumerate(labels):
                occurrences.setdefault(label, []).append((vertex, slot))

        twin_map: dict[HalfEdge, HalfEdge] = {}
        edge_label_pairs: EdgeLabelPairs = {}
        for label, half_edges in occurrences.items():
            if len(half_edges) != 2:
                label_text = QuadrangulationEnumerator._format_edge_name_for_error(
                    label
                )
                raise ValueError(
                    f"double_code edge label count invalid: {label_text!r} -> {len(half_edges)}"
                )
            first, second = half_edges
            twin_map[first] = second
            twin_map[second] = first
            edge_label_pairs[label] = (first, second)

        adjacency = {
            vertex: [
                twin_map[(vertex, slot)][0]
                for slot in range(len(labels))
            ]
            for vertex, labels in enumerate(edge_lists, start=1)
        }

        return adjacency, twin_map, edge_label_pairs
