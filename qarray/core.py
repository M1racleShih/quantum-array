from __future__ import annotations

import re
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

from .errors import QArrayConfigError, QArrayIndexError, QArrayValueError

IndexLike = Union[int, slice, Iterable[int], np.ndarray]
QubitPairs = Tuple[Tuple[int, int], Tuple[int, int]]


class TopologyType(str, Enum):
    RECTANGLE = "rect"
    RHOMBUS = "rhombus"
    BRICK = "brick6"


class QArray:
    """
    Quantum chip logical 2D layout.

    - Qubits arranged in a rows x cols grid, row-major.
    - Qubit labels: 'qK' where K uses index_base (0 or 1).
    - Coupler labels: 'cA-B' where A,B are qubit indices with same base.

    Parameters
    ----------
    rows : int
        Number of rows of qubits.
    cols : int
        Number of columns of qubits.
    index_base : int, optional
        0 or 1. If 1, top-left qubit is 'q1'; if 0, 'q0'.
    topology : str, optional
        Connectivity topology for couplers, currently supports:
        - "rect"   : rectangular 4-neighbor grid.
        - "rhombus": rhombus topology between adjacent rows. (4-neighbor)
        - "brick6" : brick-block-6-slopy topology (slanted hexagon, 3-neighbor)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        index_base: int = 1,
        topology: str = "rect",
    ):
        if rows <= 0 or cols <= 0:
            raise QArrayConfigError("rows and cols must be positive integers.")
        if index_base not in (0, 1):
            raise QArrayConfigError("index_base must be either 0 or 1.")
        if topology not in ("rect", "rhombus", "brick6"):
            raise QArrayConfigError("topology must be 'rect', 'rhombus', or 'brick6'.")

        self.rows = int(rows)
        self.cols = int(cols)
        self.n = self.rows * self.cols
        self.base = int(index_base)
        self.topology = topology

        # 2D grid of linear indices, shape (rows, cols)
        self._grid = np.arange(self.n, dtype=np.int64).reshape(self.rows, self.cols)

        # Couplers array
        self._c_array = self._build_coupler_array()

    @property
    def couplers(self):
        return self._c_array

    def __repr__(self) -> str:
        return (
            f"QArray(rows={self.rows}, cols={self.cols}, "
            f"start index={self.base}, topology='{self.topology}')"
        )

    def idx_to_rc(self, idx: int) -> Tuple[int, int]:
        """Convert a linear index (0-based) to (row, col)."""
        if idx < 0 or idx >= self.n:
            raise QArrayIndexError(f"Linear index out of range: {idx}")
        r = idx // self.cols
        c = idx % self.cols
        return r, c

    def _rc_to_idx_scalar(self, r: int, c: int) -> int:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise QArrayIndexError(f"Row/column out of range: ({r}, {c})")
        return r * self.cols + c

    def rc_to_idx(self, r, c):
        """
        Convert (row, col) to linear index.

        - If both r, c are scalars -> return int.
        - If both are ndarrays with same shape -> return ndarray.
        This function does not broadcast scalar with array; broadcasting
        for 1D row/column access is handled explicitly in __getitem__.
        """
        # scalar case
        if np.isscalar(r) and np.isscalar(c):
            return self._rc_to_idx_scalar(int(r), int(c))

        # array case: require same shape
        r_arr = np.asarray(r, dtype=np.int64)
        c_arr = np.asarray(c, dtype=np.int64)
        if r_arr.shape != c_arr.shape:
            raise QArrayIndexError(
                f"Row and column arrays must have the same shape: "
                f"{r_arr.shape} vs {c_arr.shape}"
            )

        if (r_arr < 0).any() or (r_arr >= self.rows).any():
            raise QArrayIndexError("Row indices out of range.")
        if (c_arr < 0).any() or (c_arr >= self.cols).any():
            raise QArrayIndexError("Column indices out of range.")

        return r_arr * self.cols + c_arr

    def idx_to_qubit(self, idx: int) -> str:
        """Convert a 0-based linear index to qubit label 'qK' under current base."""
        if idx < 0 or idx >= self.n:
            raise QArrayIndexError(f"Linear index out of range: {idx}")
        return f"q{idx + self.base}"

    def qubit_to_idx(self, label: str) -> int:
        """Convert label like 'q5' (base-aware) into linear index (0-based)."""
        m = re.fullmatch(r"q(\d+)", label.strip())
        if not m:
            raise QArrayValueError(f"Invalid qubit label: {label}")
        k = int(m.group(1)) - self.base
        if k < 0 or k >= self.n:
            raise QArrayIndexError(f"Qubit label out of range: {label}")
        return k

    def qubit_to_rc(self, label: str) -> Tuple[int, int]:
        """Map a qubit label like 'q5' to its (row, col) coordinates."""
        idx = self.qubit_to_idx(label)
        return self.idx_to_rc(idx)

    def _normalize_index_1d(
        self,
        idx: IndexLike,
        size: int,
        axis_name: str,
    ) -> np.ndarray:
        """
        Normalize a row or column index into an integer ndarray (1D).
        Supports int, slice, iterable of int, np.ndarray of int/bool.
        """
        if isinstance(idx, int):
            idx = idx if idx >= 0 else size + idx
            if not (0 <= idx < size):
                raise QArrayIndexError(
                    f"{axis_name} index out of range: {idx} (size={size})"
                )
            return np.array([idx], dtype=np.int64)

        if isinstance(idx, slice):
            rng = np.arange(size)[idx]
            return rng.astype(np.int64)

        if isinstance(idx, np.ndarray):
            arr = idx
            if arr.dtype == bool:
                if arr.size != size:
                    raise QArrayIndexError(
                        f"Boolean index for {axis_name} has wrong length: "
                        f"{arr.size}, expected {size}"
                    )
                return np.nonzero(arr)[0].astype(np.int64)
            arr = arr.astype(np.int64, copy=False)
            arr = np.where(arr < 0, arr + size, arr)
            if (arr < 0).any() or (arr >= size).any():
                raise QArrayIndexError(
                    f"{axis_name} index contains value out of range."
                )
            return arr

        if isinstance(idx, Iterable):
            arr = np.fromiter((int(x) for x in idx), dtype=np.int64)
            arr = np.where(arr < 0, arr + size, arr)
            if (arr < 0).any() or (arr >= size).any():
                raise QArrayIndexError(
                    f"{axis_name} index contains value out of range."
                )
            return arr

        raise QArrayIndexError(f"Unsupported index type for {axis_name}: {type(idx)}")

    def __getitem__(self, key):
        """
        Numpy-like indexing that returns qubit labels.

        Rules
        -----
        - A[r, c] with both int -> single string label 'qK'.
        - A[r, :] or A[:, c] or A[r, [..]] or A[[..], c]
          -> 1D np.ndarray of labels.
        - A[rows, cols] (both arrays/slices)
          -> 2D np.ndarray of labels.
        - A[k] (1D / linear indexing) -> 1D array (or scalar) of labels.
        """
        # 1D / linear indexing
        if not isinstance(key, tuple):
            idx_arr = self._normalize_index_1d(key, self.n, "linear")
            labels = np.array(
                [self.idx_to_qubit(int(i)) for i in idx_arr],
                dtype=object,
            )
            # preserve scalar when key is pure int
            if np.isscalar(key):
                return labels[0]
            return labels

        # 2D indexing
        if len(key) != 2:
            raise QArrayIndexError("Only 2D indexing is supported for QArray.")

        r_key, c_key = key
        r_idx = self._normalize_index_1d(r_key, self.rows, "row")
        c_idx = self._normalize_index_1d(c_key, self.cols, "col")

        # scalar -> single label
        if r_idx.size == 1 and c_idx.size == 1:
            idx = self._rc_to_idx_scalar(int(r_idx[0]), int(c_idx[0]))
            return self.idx_to_qubit(int(idx))

        # 1D row: shape (len(c_idx),)
        if r_idx.size == 1 and c_idx.size > 1:
            r = int(r_idx[0])
            lin = r * self.cols + c_idx
            labels = np.array(
                [self.idx_to_qubit(int(i)) for i in lin],
                dtype=object,
            )
            return labels

        # 1D column: shape (len(r_idx),)
        if r_idx.size > 1 and c_idx.size == 1:
            c = int(c_idx[0])
            lin = r_idx * self.cols + c
            labels = np.array(
                [self.idx_to_qubit(int(i)) for i in lin],
                dtype=object,
            )
            return labels

        # 2D subarray: meshgrid then map to labels
        rr, cc = np.meshgrid(r_idx, c_idx, indexing="ij")
        lin = rr * self.cols + cc
        labels = np.vectorize(
            lambda k: self.idx_to_qubit(int(k)),
            otypes=[object],
        )(lin)
        return labels

    @staticmethod
    def _coupler_label_from_indices(a: int, b: int, base: int) -> str:
        """
        Build a normalized coupler label from two linear indices (0-based).

        The endpoints are ordered so that the smaller physical index (with base
        applied) comes first, thus 'c1-7' and 'c7-1' are the same coupler.
        """
        a_phys = a + base
        b_phys = b + base
        if a_phys <= b_phys:
            return f"c{a_phys}-{b_phys}"
        else:
            return f"c{b_phys}-{a_phys}"

    def _build_coupler_array(self) -> BaseCouplerArray:
        """
        Build a topology-specific CouplerArray view over this QArray.
        """
        if self.topology == "rect":
            return RectCouplerArray(self)
        if self.topology == "rhombus":
            return RhombusCouplerArray(self)
        raise QArrayConfigError(
            f"Unsupported topology: {self.topology}, only support:\n rect, rhombus, brick6"
        )

    def couplers_all(self) -> List[str]:
        """Return all coupler labels under the current topology."""
        return self.couplers.labels_all()

    def couplers_of(self, q_label: str) -> List[str]:
        """
        Return all coupler labels that connect to a given qubit label,
        under the current topology.
        """
        return self.couplers.couplers_of(q_label)


class BaseCouplerArray:
    """
    Abstract coupler array view built on top of a QArray.

    Provides:
    - Access by (row, col) through numpy-style indexing (in subclasses).
    - Conversion to / from coupler labels 'cA-B'.
    """

    def __init__(self, qarray: QArray):
        self.q = qarray

    def __getitem__(self, key):
        raise NotImplementedError

    def labels_all(self) -> List[str]:
        raise NotImplementedError

    def get_loc(self, c_label: str) -> Union[Tuple[int, int, int], None]:
        raise NotImplementedError

    def at(self, r: int, c: int) -> List[str]:
        raise NotImplementedError

    def get_qubits_rc(self, c_label: str) -> QubitPairs:
        raise NotImplementedError

    def couplers_of(self, q_label: str) -> List[str]:
        raise NotImplementedError


class RectCouplerArray(BaseCouplerArray):
    """
    Numpy-style Matrix implementation for Rectangular topology.

    Storage Structure: (rows, cols, 2)
    - Channel 0: Horizontal (Right) -> Connects (r, c) with (r, c+1)
    - Channel 1: Vertical   (Down)  -> Connects (r, c) with (r+1, c)
    """

    def __init__(self, qarray: QArray):
        self.q = qarray
        if self.q.topology != "rect":
            raise Exception("RectCouplerGrid requires 'rect' topology QArray.")

        self.rows = self.q.rows
        self.cols = self.q.cols

        # 1. Initialize grid with None
        # Shape: (Rows, Cols, 2)
        self._data = np.full((self.rows, self.cols, 2), None, dtype=object)

        # 2. Initialize Reverse Lookup Map
        # Maps 'c1-2' -> (row, col, channel)
        self._label_to_loc: Dict[str, Tuple[int, int, int]] = {}

        self._build_matrix()

    def _build_matrix(self):
        grid = self.q._grid
        base = self.q.base

        # 1. Horizontal (Channel 0)
        if self.cols >= 2:
            src_idx = grid[:, :-1]
            dst_idx = grid[:, 1:]
            labels = np.vectorize(
                lambda a, b: self.q._coupler_label_from_indices(a, b, base)
            )(src_idx, dst_idx)
            self._data[:, :-1, 0] = labels

        # 2. Vertical (Channel 1)
        if self.rows >= 2:
            src_idx = grid[:-1, :]
            dst_idx = grid[1:, :]
            labels = np.vectorize(
                lambda a, b: self.q._coupler_label_from_indices(a, b, base)
            )(src_idx, dst_idx)
            self._data[:-1, :, 1] = labels

        # Build Reverse Map (Inverse Lookup)
        # Iterate through the populated matrix to build the dictionary
        # This ensures O(1) lookup speed for get_loc
        it = np.nditer(self._data, flags=["multi_index", "refs_ok"])
        for x in it:
            val = x.item()
            if val is not None:
                # multi_index returns (r, c, k)
                self._label_to_loc[val] = it.multi_index

    def __repr__(self):
        # Simply show the data array representation
        return repr(self._data)

    def __getitem__(self, key):
        # TODO, filter out None values
        return self._data[key]

    def get_loc(self, c_label: str) -> Union[Tuple[int, int, int], None]:
        """
        Get the (row, col, channel) index for a given coupler label.
        Returns None if the coupler does not exist in the matrix.

        Channel 0 = Horizontal (Right)
        Channel 1 = Vertical (Down)
        """
        return self._label_to_loc.get(c_label)

    def at(self, r: int, c: int) -> List[str]:
        """Return all valid downward couplers starting from (r, c)."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise QArrayIndexError(f"Index ({r}, {c}) out of range.")

        couplers = self._data[r, c]  # Shape (2,)
        return [x for x in couplers if x is not None]

    def couplers_of(self, q_label: str) -> List[str]:
        idx = self.q.qubit_to_idx(q_label)
        r, c = self.q.idx_to_rc(idx)
        res = []

        # Outgoing (Source)
        v_right = self._data[r, c, 0]
        v_down = self._data[r, c, 1]
        if v_right:
            res.append(v_right)
        if v_down:
            res.append(v_down)

        # Incoming (Target)
        if c > 0:
            v_left = self._data[r, c - 1, 0]
            if v_left:
                res.append(v_left)
        if r > 0:
            v_up = self._data[r - 1, c, 1]
            if v_up:
                res.append(v_up)

        return sorted(res)

    def labels_all(self) -> List[str]:
        return list(self._label_to_loc.keys())

    def get_qubits_rc(self, c_label: str) -> QubitPairs:
        m = re.fullmatch(r"c(\d+)-(\d+)", c_label.strip())
        if not m:
            raise QArrayValueError(f"Invalid coupler label: {c_label}")

        a_lin = int(m.group(1)) - self.q.base
        b_lin = int(m.group(2)) - self.q.base

        for k in (a_lin, b_lin):
            if k < 0 or k >= self.q.n:
                raise QArrayIndexError(f"Coupler endpoint out of range: {c_label}")

        qa = self.q.idx_to_qubit(a_lin)
        qb = self.q.idx_to_qubit(b_lin)

        ra, ca = self.q.idx_to_rc(a_lin)
        rb, cb = self.q.idx_to_rc(b_lin)

        if abs(ra - rb) + abs(ca - cb) != 1:
            raise QArrayValueError(
                f"Non-adjacent qubits cannot form a rectangular coupler: {qa}, {qb}"
            )

        return (ra, ca), (rb, cb)


class RhombusCouplerArray(BaseCouplerArray):
    """
    A Numpy-style matrix representation for Rhombus topology couplers.

    Storage Structure
    Shape: (rows, cols, 2)
    - channel 0: Vertical coupler starting from (r, c) downwards.
    - channel 1: Diagonal coupler starting from (r, c) downwards.
        - If r is Even: Connects to (r+1, c-1) [Left-Down]
        - If r is Odd : Connects to (r+1, c+1) [Right-Down]

    This allows accessing couplers via C[row, col].
    """

    def __init__(self, qarray: QArray):
        self.q = qarray
        if self.q.topology != "rhombus":
            raise QArrayConfigError("CouplerGrid requires Rhombus topology QArray.")

        self.rows = self.q.rows
        self.cols = self.q.cols

        # Initialize grid with None (or empty string '')
        # Shape: (Rows, Cols, 2 types)
        self._data = np.full((self.rows, self.cols, 2), None, dtype=object)

        # Pre-compute indices mapping (Inverse map)
        # Maps 'cA-B' -> (r, c, type_idx)
        self._label_to_loc: Dict[str, Tuple[int, int, int]] = {}

        self._build_matrix()

    def _build_matrix(self):
        """Populate the (R, C, 2) matrix."""
        grid = self.q._grid
        base = self.q.base

        # 1. Fill Vertical (Channel 0)
        # Valid for rows 0 to R-2
        if self.rows >= 2:
            # Sources: (r, c), Targets: (r+1, c)
            src_idx = grid[0 : self.rows - 1, :]
            dst_idx = grid[1 : self.rows, :]

            # Vectorized label creation
            labels = np.vectorize(
                lambda a, b: self.q._coupler_label_from_indices(a, b, base)
            )(src_idx, dst_idx)

            self._data[0 : self.rows - 1, :, 0] = labels

        # 2. Fill Diagonal (Channel 1)
        # Valid for rows 0 to R-2
        for r in range(self.rows - 1):
            # Even Row: (r, c) -> (r+1, c-1)
            if r % 2 == 0:
                # Valid columns: 1 to C-1 (since c-1 must be >= 0)
                if self.cols > 1:
                    src_idx = grid[r, 1:]  # (r, 1) ... (r, last)
                    dst_idx = grid[r + 1, :-1]  # (r+1, 0) ... (r+1, last-1)
                    labels = np.vectorize(
                        lambda a, b: self.q._coupler_label_from_indices(a, b, base)
                    )(src_idx, dst_idx)
                    self._data[r, 1:, 1] = labels

            # Odd Row: (r, c) -> (r+1, c+1)
            else:
                # Valid columns: 0 to C-2 (since c+1 must be < C)
                if self.cols > 1:
                    src_idx = grid[r, :-1]  # (r, 0) ... (r, last-1)
                    dst_idx = grid[r + 1, 1:]  # (r+1, 1) ... (r+1, last)
                    labels = np.vectorize(
                        lambda a, b: self.q._coupler_label_from_indices(a, b, base)
                    )(src_idx, dst_idx)
                    self._data[r, :-1, 1] = labels

        # Build reverse map
        it = np.nditer(self._data, flags=["multi_index", "refs_ok"])
        for x in it:
            val = x.item()
            if val is not None:
                self._label_to_loc[val] = it.multi_index

    def __getitem__(self, key):
        """
        Support numpy slicing.
        Returns a ndarray of coupler strings (or None).
        """
        # TODO, filter out None values
        return self._data[key]

    def __repr__(self):
        # Simply show the data array representation
        return repr(self._data)

    def get_loc(self, c_label: str) -> Union[Tuple[int, int, int], None]:
        """
        Get the (row, col, channel) index for a given coupler label.
        channel 0 = Vertical, 1 = Diagonal
        """
        return self._label_to_loc.get(c_label)

    def at(self, r: int, c: int) -> List[str]:
        """Return all valid downward couplers starting from (r, c)."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise QArrayIndexError(f"Index ({r}, {c}) out of range.")

        couplers = self._data[r, c]  # Shape (2,)
        return [x for x in couplers if x is not None]

    def couplers_of(self, q_label: str) -> List[str]:
        """
        Get all couplers connected to a specific qubit using the matrix lookup.
        """
        idx = self.q.qubit_to_idx(q_label)
        r, c = self.q.idx_to_rc(idx)

        found_couplers = []

        # 1. Downward Connections (Source = Self)
        # Direct lookup: self[r, c, :]
        if r < self.rows:
            # Channel 0: Vertical Down
            val = self._data[r, c, 0]
            if val is not None:
                found_couplers.append(val)

            # Channel 1: Diagonal Down
            val = self._data[r, c, 1]
            if val is not None:
                found_couplers.append(val)

        # 2. Upward Connections (Target = Self)
        # The table for the previous row needs to be looked up: self[r-1, ?, :]
        if r > 0:
            prev_r = r - 1

            # A. Vertical Up
            # Source is directly above: (r-1, c)
            val = self._data[prev_r, c, 0]
            if val is not None:
                found_couplers.append(val)

            # B. Diagonal Up
            # We need to inverse the logic of Diagonal Down.
            # Logic:
            # - If prev_r is Even ('/'): It connects (prev_r, k) -> (r, k-1).
            #   We are at c, so c = k-1 => k = c+1. Source is (prev_r, c+1).
            # - If prev_r is Odd  ('\'): It connects (prev_r, k) -> (r, k+1).
            #   We are at c, so c = k+1 => k = c-1. Source is (prev_r, c-1).

            src_col = -1
            if prev_r % 2 == 0:
                src_col = c + 1  # Look Top-Right
            else:
                src_col = c - 1  # Look Top-Left

            # Check bounds for the source column
            if 0 <= src_col < self.cols:
                val = self._data[prev_r, src_col, 1]  # Channel 1 is always diagonal
                if val is not None:
                    found_couplers.append(val)

        return sorted(found_couplers)

    def get_qubits_rc(self, c_label: str) -> QubitPairs:
        m = re.fullmatch(r"c(\d+)-(\d+)", c_label.strip())
        if not m:
            raise QArrayValueError(f"Invalid coupler label: {c_label}")

        a_lin = int(m.group(1)) - self.q.base
        b_lin = int(m.group(2)) - self.q.base

        for k in (a_lin, b_lin):
            if k < 0 or k >= self.q.n:
                raise QArrayIndexError(f"Coupler endpoint out of range: {c_label}")

        qa = self.q.idx_to_qubit(a_lin)
        qb = self.q.idx_to_qubit(b_lin)

        ra, ca = self.q.idx_to_rc(a_lin)
        rb, cb = self.q.idx_to_rc(b_lin)

        dr = rb - ra
        dc = cb - ca

        if not (
            (dr == 1 and dc == 0)
            or (dr == -1 and dc == 0)
            or (dr == 1 and dc == 1)
            or (dr == -1 and dc == 1)
        ):
            raise QArrayValueError(
                f"Endpoints do not form a valid rhombus coupler: {qa}, {qb}"
            )

        return (ra, ca), (rb, cb)

    def labels_all(self) -> List[str]:
        return list(self._label_to_loc.keys())
