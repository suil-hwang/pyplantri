# src/pyplantri/plantri.py
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from shutil import which
from collections.abc import Iterator
from typing import Literal

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


def _is_ascii_digit_byte(value: int) -> bool:
    """Returns True if value is an ASCII digit byte ('0'..'9')."""
    return 48 <= value <= 57


def _token_starts_with_digit(token: str | bytes) -> bool:
    """Returns True when a token begins with an ASCII digit."""
    head = token[0]
    if isinstance(head, str):
        return head.isdigit()
    return _is_ascii_digit_byte(head)


def _iter_edge_labels(edge_labels: str | bytes) -> Iterator[EdgeLabel]:
    """Iterate edge labels from one plantri -T token."""
    if isinstance(edge_labels, bytes):
        return iter(edge_labels)
    return iter(edge_labels)


def _find_plantri_exe() -> Path:
    """Finds the plantri executable path."""
    exe_name = "plantri.exe" if os.name == "nt" else "plantri"

    # Package bin folder (installed).
    pkg_bin = Path(__file__).parent / "bin" / exe_name
    if pkg_bin.exists():
        return pkg_bin

    # scikit-build-core build folder (editable install, dev mode).
    # Path: src/pyplantri/plantri.py -> src/pyplantri -> src -> project_root.
    project_root = Path(__file__).parent.parent.parent
    build_dir = project_root / "build"
    if build_dir.exists():
        for tag_dir in build_dir.iterdir():
            if tag_dir.is_dir():
                # Release folder (Visual Studio build).
                release_exe = tag_dir / "Release" / exe_name
                if release_exe.exists():
                    return release_exe
                # MinGW/Unix build.
                direct_exe = tag_dir / exe_name
                if direct_exe.exists():
                    return direct_exe

    path_exe = which(exe_name)
    if path_exe is not None:
        return Path(path_exe)

    # Default path for error messages.
    return Path(__file__).parent / "bin" / exe_name


_PLANTRI_EXE = _find_plantri_exe()
_BUNDLED_PLANTRI_MAX_PRIMAL_VERTICES = 64
_BUNDLED_MAX_DUAL_VERTEX_COUNT = _BUNDLED_PLANTRI_MAX_PRIMAL_VERTICES - 2


class PlantriError(Exception):
    """Plantri execution failure."""


class PlantriExecutableNotFoundError(PlantriError, FileNotFoundError):
    """Plantri executable could not be found."""


class QuadrangulationDualClass(str, Enum):
    """Dual graph classes available from plantri quadrangulation modes."""

    QUARTIC_MULTIGRAPH = "quartic_multigraph"
    SIMPLE_QUARTIC = "simple_quartic"


def _raise_executable_not_found(executable: Path) -> None:
    raise PlantriExecutableNotFoundError(
        f"plantri: executable not found {executable}; "
        "run 'pip install -e .' or add plantri to PATH"
    )


class Plantri:
    """Wrapper for the plantri executable."""
    _COUNT_INCOMPATIBLE_OUTPUT_OPTIONS = frozenset({"-a", "-g", "-s", "-E", "-T", "-u"})

    def __init__(self, executable: Path | None = None) -> None:
        """Initializes Plantri with the executable path."""
        self.executable = Path(executable) if executable else _PLANTRI_EXE
        if not self.executable.exists():
            _raise_executable_not_found(self.executable)

    def run(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> bytes:
        """Runs plantri with the given parameters."""
        cmd = self._build_command(
            n_vertices,
            options=options,
            output_format=output_format,
        )

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors="replace") if e.stderr else str(e)
            raise PlantriError(
                f"plantri: execution failed (exit {e.returncode}); "
                f"{_summarize_process_text(stderr_text)}"
            ) from e
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(
                f"plantri: executable not found {self.executable}"
            ) from e

    def _build_command(
        self,
        n_vertices: int,
        *,
        options: list[str] | None,
        output_format: Literal["planar_code", "ascii"],
    ) -> list[str]:
        """Builds a plantri command line for the given options."""
        if output_format not in ("planar_code", "ascii"):
            raise ValueError(
                f"plantri: unsupported output_format {output_format!r}; "
                "use 'planar_code' or 'ascii'"
            )

        cmd = [str(self.executable)]
        if options:
            cmd.extend(options)

        # Set output format flag.
        if output_format == "ascii" and "-a" not in (options or []):
            cmd.append("-a")

        cmd.append(str(n_vertices))
        return cmd

    def iter_stdout_lines(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> Iterator[bytes]:
        """Stream non-empty stdout lines for line-oriented plantri output."""
        cmd = self._build_command(
            n_vertices,
            options=options,
            output_format=output_format,
        )

        with tempfile.TemporaryFile() as stderr_file:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                )
            except FileNotFoundError as e:
                raise PlantriExecutableNotFoundError(
                    f"plantri: executable not found {self.executable}"
                ) from e

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

                if fully_consumed:
                    return_code = proc.wait()
                else:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                    return_code = proc.returncode if proc.returncode is not None else 0

                stderr_file.seek(0)
                stderr_text = stderr_file.read().decode("utf-8", errors="replace")
                if fully_consumed and return_code != 0:
                    stderr_excerpt = _summarize_process_text(stderr_text, limit=4000)
                    raise PlantriError(
                        f"plantri: execution failed (exit {return_code}); "
                        f"{stderr_excerpt}"
                    )

    def count_from_options(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        timeout: float = 3600.0,
    ) -> int:
        """Counts graphs with arbitrary generation options via plantri ``-u``."""
        normalized_options = [
            opt
            for opt in (options or [])
            if opt not in self._COUNT_INCOMPATIBLE_OUTPUT_OPTIONS
        ]

        try:
            result = subprocess.run(
                [str(self.executable)] + normalized_options + ["-u", str(n_vertices)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Check exit code
            if result.returncode != 0:
                stderr_excerpt = _summarize_process_text(result.stderr)
                stdout_excerpt = _summarize_process_text(result.stdout)
                raise PlantriError(
                    f"plantri: count failed (exit {result.returncode}); "
                    f"stderr={stderr_excerpt}; stdout={stdout_excerpt}"
                )

            # Parse count from stderr (e.g., "1 graphs written to stdout" or "1 quadrangulations generated").
            for line in result.stderr.split("\n"):
                match = re.search(
                    r"(\d+)\s+.*\b(?:graph|triangulation|quadrangulation)s?\b",
                    line.lower(),
                )
                if match:
                    return int(match.group(1))

            # If we reach here, parsing failed
            stderr_excerpt = _summarize_process_text(result.stderr)
            stdout_excerpt = _summarize_process_text(result.stdout)
            raise PlantriError(
                "plantri: count parse failed; "
                f"stderr={stderr_excerpt}; stdout={stdout_excerpt}"
            )

        except subprocess.TimeoutExpired as e:
            raise PlantriError(
                f"plantri: timed out after {timeout}s for n={n_vertices}, "
                f"options={normalized_options}"
            ) from e

        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(
                f"plantri: executable not found {self.executable}"
            ) from e

        except PlantriError:
            # Re-raise our own errors
            raise

        except Exception as e:
            # Catch-all for unexpected errors
            raise PlantriError(
                f"plantri: unexpected {type(e).__name__}: {e}"
            ) from e


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
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: 3,
        QuadrangulationDualClass.SIMPLE_QUARTIC: 6,
    }

    def __init__(self) -> None:
        """Initializes the SQS enumerator with a Plantri instance."""
        self._plantri: Plantri | None = None

    def _get_plantri(self) -> Plantri:
        if self._plantri is None:
            self._plantri = Plantri()
        return self._plantri

    @classmethod
    def _normalize_dual_class(
        cls,
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
        dual_class: QuadrangulationDualClass | str,
    ) -> list[str]:
        resolved_dual_class = cls._normalize_dual_class(dual_class)
        return list(cls._FLAGS_BY_DUAL_CLASS[resolved_dual_class])

    @classmethod
    def _dual_class_from_filter(
        cls,
        *,
        double_edge_free_only: bool,
    ) -> QuadrangulationDualClass:
        if double_edge_free_only:
            return QuadrangulationDualClass.SIMPLE_QUARTIC
        return QuadrangulationDualClass.QUARTIC_MULTIGRAPH

    @classmethod
    def _min_nonempty_dual_vertices(
        cls,
        dual_class: QuadrangulationDualClass | str,
    ) -> int:
        resolved_dual_class = cls._normalize_dual_class(dual_class)
        return cls._MIN_NONEMPTY_DUAL_VERTICES[resolved_dual_class]

    @staticmethod
    def _validate_supported_dual_vertex_count(dual_vertex_count: int) -> None:
        """Reject dual sizes outside the bundled plantri build range."""
        if dual_vertex_count < 3:
            raise ValueError(
                f"dual_vertex_count unsupported: {dual_vertex_count} < 3"
            )
        if dual_vertex_count > _BUNDLED_MAX_DUAL_VERTEX_COUNT:
            raise ValueError(
                "dual_vertex_count unsupported: "
                f"{dual_vertex_count} > {_BUNDLED_MAX_DUAL_VERTEX_COUNT} "
                f"(bundled plantri MAXN={_BUNDLED_PLANTRI_MAX_PRIMAL_VERTICES})"
            )

    def generate_pairs(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> Iterator[tuple[ParsedGraphSection, ParsedGraphSection]]:
        """Yield (primal, dual) pairs from plantri."""
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        for line in self.iter_double_code_lines(
            dual_vertex_count,
            dual_class=dual_class,
        ):
            yield self.parse_double_code(line)

    def count(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> int:
        """Count non-isomorphic duals of simple quadrangulations."""
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
        dual_class: QuadrangulationDualClass = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> Iterator[bytes]:
        """Yield raw double_code lines as bytes from plantri stdout."""
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = self._normalize_dual_class(dual_class)
        if dual_vertex_count < self._min_nonempty_dual_vertices(resolved_dual_class):
            return
        # Euler's formula for plane graphs: V - E + F = 2
        # For quadrangulations: primal_vertices = dual_vertices + 2
        primal_vertex_count = dual_vertex_count + 2
        for line in self._get_plantri().iter_stdout_lines(
            primal_vertex_count,
            self._flags_for_dual_class(resolved_dual_class),
        ):
            if _token_starts_with_digit(line):
                yield line

    @staticmethod
    def parse_double_code(
        double_code_line: str | bytes,
    ) -> tuple[ParsedGraphSection, ParsedGraphSection]:
        """Parse a plantri double_code line into (primal, dual) sections.

        Without -d, plantri outputs primal first then dual. With -d
        the order is reversed. This method detects the orientation via
        4-regularity and vertex-count checks.
        """
        parts = list(double_code_line.split())
        first_vertex_count, first_edge_lists, next_idx = (
            QuadrangulationEnumerator._parse_section(parts, 0, "first")
        )
        second_vertex_count, second_edge_lists, next_idx = (
            QuadrangulationEnumerator._parse_section(parts, next_idx, "second")
        )
        if next_idx != len(parts):
            raise ValueError(
                f"double_code trailing token count: {len(parts) - next_idx}"
            )

        first_data = QuadrangulationEnumerator._build_section(
            first_vertex_count,
            first_edge_lists,
        )
        second_data = QuadrangulationEnumerator._build_section(
            second_vertex_count,
            second_edge_lists,
        )
        return QuadrangulationEnumerator._resolve_primal_dual_sections(
            first_data,
            second_data,
        )

    @staticmethod
    def _parse_section(
        parts: list[str | bytes],
        start_idx: int,
        section_name: str,
    ) -> tuple[int, list[str | bytes], int]:
        """Parse one double_code section header and its edge-label tokens."""
        if start_idx >= len(parts):
            raise ValueError(f"double_code missing {section_name} section header")

        vertex_count = int(parts[start_idx])
        if vertex_count < 0:
            raise ValueError(f"double_code {section_name} count invalid: {vertex_count}")
        idx = start_idx + 1
        end_idx = idx + vertex_count
        if end_idx > len(parts):
            raise ValueError(
                f"double_code {section_name} count mismatch: {len(parts) - idx} != {vertex_count}"
            )
        edge_lists = parts[idx:end_idx]

        return vertex_count, edge_lists, end_idx

    @staticmethod
    def _build_section(
        vertex_count: int,
        edge_lists: list[str | bytes],
    ) -> ParsedGraphSection:
        """Build one parsed section from edge-label token lists."""
        adjacency, twin_map, edge_label_pairs = (
            QuadrangulationEnumerator._build_adjacency_and_twins(edge_lists)
        )
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
        QuadrangulationEnumerator._validate_cross_section_edge_labels(
            first_data,
            second_data,
        )

        first_is_4_regular = GraphConverter.is_4_regular(first_data.cyclic_adjacency)
        second_is_4_regular = GraphConverter.is_4_regular(second_data.cyclic_adjacency)

        if first_is_4_regular == second_is_4_regular:
            raise ValueError(
                f"double_code quartic classification invalid: ({first_is_4_regular}, {second_is_4_regular})"
            )

        if first_is_4_regular:
            dual_data, primal_data = first_data, second_data
        else:
            dual_data, primal_data = second_data, first_data

        if primal_data.vertex_count != dual_data.vertex_count + 2:
            raise ValueError(
                f"double_code primal/dual vertex mismatch: primal={primal_data.vertex_count}, dual={dual_data.vertex_count}"
            )

        return primal_data, dual_data

    @staticmethod
    def _validate_cross_section_edge_labels(
        first_data: ParsedGraphSection,
        second_data: ParsedGraphSection,
    ) -> None:
        """Check that both sections describe the same labeled edge set."""
        first_edge_count = len(first_data.edge_label_pairs)
        second_edge_count = len(second_data.edge_label_pairs)
        if first_edge_count != second_edge_count:
            raise ValueError(
                f"double_code edge count mismatch: {first_edge_count} != {second_edge_count}"
            )

        first_labels = set(first_data.edge_label_pairs)
        second_labels = set(second_data.edge_label_pairs)
        if first_labels != second_labels:
            missing_in_second = sorted(
                (
                    QuadrangulationEnumerator._format_edge_name_for_error(label)
                    for label in first_labels - second_labels
                )
            )
            missing_in_first = sorted(
                (
                    QuadrangulationEnumerator._format_edge_name_for_error(label)
                    for label in second_labels - first_labels
                )
            )
            raise ValueError(
                f"double_code edge label mismatch: first-only={missing_in_second}, second-only={missing_in_first}"
            )

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
        edge_lists: list[str | bytes],
    ) -> tuple[
        dict[int, list[int]],
        dict[HalfEdge, HalfEdge],
        EdgeLabelPairs,
    ]:
        """Build adjacency, twin map, and edge-label/half-edge pairs."""
        # Collect (vertex, position) pairs where each edge name appears.
        edge_name_to_half_edges: dict[EdgeLabel, list[HalfEdge]] = {}
        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            for pos, edge_name in enumerate(_iter_edge_labels(edges_str)):
                edge_name_to_half_edges.setdefault(edge_name, []).append(
                    (vertex_idx, pos)
                )

        # Build adjacency list.
        adjacency: dict[int, list[int]] = {}
        for vertex_idx, edges_str in enumerate(edge_lists, start=1):
            neighbors: list[int] = []
            for edge_name in _iter_edge_labels(edges_str):
                half_edges = edge_name_to_half_edges.get(edge_name)
                if half_edges is None or len(half_edges) != 2:
                    edge_name_str = QuadrangulationEnumerator._format_edge_name_for_error(
                        edge_name
                    )
                    raise ValueError(
                        f"double_code edge label count invalid: {edge_name_str!r} -> {0 if half_edges is None else len(half_edges)}"
                    )
                (v1, _), (v2, _) = half_edges
                if v1 == v2 == vertex_idx:
                    neighbors.append(vertex_idx)  # Loop edge
                else:
                    neighbors.append(v2 if v1 == vertex_idx else v1)
            adjacency[vertex_idx] = neighbors

        # Twin mapping: match two half-edges sharing the same edge name.
        twin_map: dict[HalfEdge, HalfEdge] = {}
        edge_label_pairs: EdgeLabelPairs = {}
        for half_edges in edge_name_to_half_edges.values():
            if len(half_edges) == 2:
                twin_map[half_edges[0]] = half_edges[1]
                twin_map[half_edges[1]] = half_edges[0]
        for edge_name, half_edges in edge_name_to_half_edges.items():
            if len(half_edges) == 2:
                edge_label_pairs[edge_name] = (half_edges[0], half_edges[1])

        return adjacency, twin_map, edge_label_pairs
