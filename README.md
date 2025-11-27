# QArray: Using qubits and couplers from an array perspective

QArray is a small Python library for **expressing and manipulating quantum chip topologies** with an array / geometry view:

> You describe the chip as a 2D array of qubits and couplers, rather than as an abstract graph of nodes and edges.

It is especially useful for:

- Regular 2D architectures (e.g. superconducting grid‑like layouts)
- Tasks that need frequent “by row / column / rectangular region” selection
- Acting as a light‑weight **hardware topology modeling layer** before Qiskit, NetworkX or other compilers

The library provides a single core abstraction:

- `QArray` – a geometry‑aware view over qubits on a 2D grid and couplers on a 3D grid.

and currently supports at least:

- Rectangular‑like 2D layouts
- Rhombus / brick‑wall like connectivity (exposed via configuration / helpers)

---

## Installation

From source:

```bash
git clone https://github.com/M1racleShih/quantum-array.git
cd quantum-array
pip install -e .
```
---

## Data Model Overview
### Qubits as a 2‑D array
A QArray describes qubits on a 2‑D grid:
```python
from qarray import QArray

qa = QArray(shape=(4, 4))   # 4 rows x 4 cols
print(qa.shape)             # (4, 4)
```
Conceptually, qubits form a 2‑D array:

- qa[row, col] gives you the qubit at that coordinate
- You can slice as in NumPy: qa[1:3, 0:2], qa[0, :], qa[:, 2], etc.


### Couplers as a 3‑D array
The representation of couplers (two-bit connections) in QArray is a key design feature of this library.

Coupler labels are stored in the form of a 3D tensor (R × C × 2).
- The first two dimensions correspond to the qubit coordinates (r, c).
- The third dimension represents the **“channel”** or **“direction”** (e.g., channel 0 denotes right/vertical, channel 1 denotes down/diagonal).
- Advantage: Couplers can be located by their coordinates (r, c) within O(1) time complexity.
- Conversely, it is also possible to look up the physical location in O(1) time using the tag (e.g., c1-2).

---

## Quick Start
### Creating a simple rhombus chip
```python
from qarray import QArray

# Create a 4x6 QArray with "rhombus" style connectivity
# (Exact constructor signature may differ slightly; adjust if needed.)
qarray =  QArray(4, 6, index_base=1, topology="rhombus")   # rows=4, cols=6

```

### Accessing qubits by indexes
QArray supports NumPy‑like indexing:
```python
# Access the qubit at row 2, column 3
q = qarray[2, 3]
print(q)         
# Implementation‑defined qubit representation, default is a string like "q16"

# You can also use Python slices
patch = qarray[1:3, 1:3]   
# A 2x2 "view" over a sub‑region of the chip
# [['q8' 'q9']
# ['q14' 'q15']]

# Row selection
row_0 = qarray[0, :]       
# All qubits in the first row
# ['q1' 'q2' 'q3' 'q4' 'q5' 'q6']

# Column selection
col_1 = qarray[:, 1]       
# All qubits in the second column
# ['q2' 'q8' 'q14' 'q20']
```
What you get back (a single qubit object, a 1D view, or a 2D view) depends on the actual implementation, but the library is designed to “feel” like working with a 2D array while still exposing qubit / coupler semantics.

### Couplers and connectivity

```python
# Get the couplers location by labels.    
lbl1 = "c1-7"
loc1 = qarray.couplers.get_loc(lbl1)
print(f"Location of {lbl1} : {loc1}")
# Expected: (0, 0, 0) -> Row 0, Col 0, Vertical Channel

lbl2 = "c3-8"
loc2 = qarray.couplers.get_loc(lbl2)
print(f"Location of {lbl2}: {loc2}")
# Expected: (0, 2, 1) -> Row 1, Col 1, Diag Channel

lbl3 = "c99-100"
loc3 = qarray.couplers.get_loc(lbl3)
print(f"Location of {lbl3}: {loc3}")
# Expected: None

# Verify consistency
if loc1:
    r, c, k = loc1
    print(f"Verify C[{r}, {c}, {k}] == {lbl1}: {C_rect[r,c,k] == lbl1}")

# couplers also support NumPy‑like indexing:

# Get all vertical couplers in the first row
print(qarray.couplers[0, :, 0])
# ['c1-7' 'c2-8' 'c3-9' 'c4-10' 'c5-11' 'c6-12']  

# Use .at(r, c) to get all valid downward couplers starting from (r, c). 
print(qarray.couplers.at(1,1))
# ['c8-14', 'c8-15']

# You can get all couplers connected to a specific qubit.

print(qarray.couplers_of("q9"))
# ['c3-9', 'c4-9', 'c9-15', 'c9-16']

```

---

## Supported Topologies
### Rectangular (default)

- Implementation Class: RectCouplerArray.
- Connection Method: Standard 4-neighbor grid (up, down, left, right).
- Storage: Channel 0 stores horizontal connections, Channel 1 stores vertical connections.

### Rhombus

- Implementation Class: RhombusCouplerArray.
- Connection Pattern: Similar to an inclined grid.
- Logic: Distinguishes between even and odd rows.
    - Channel 0: Vertically downward.
    - Channel 1: Diagonally downward (even rows toward the lower-left, odd rows toward the lower-right).

---

## Why QArray?
### 1. Geometry‑first view of the chip

In many tools, a quantum chip is modeled purely as a graph:

- Nodes: qubits
- Edges: couplers

Graph libraries such as NetworkX are great at running algorithms (shortest path, cuts, clustering), but they do not know anything about rows, columns, coordinates, or rectangular regions unless you maintain that yourself as extra attributes.

QArray takes the opposite standpoint:

- The chip is first and foremost a 2D array with (row, col) coordinates
- The connectivity pattern (rectangular, rhombus, etc.) is derived from the geometry
- Indexing and slicing follow a NumPy‑like style

This is much more natural for:

- Superconducting grid‑style chips
- Experimenting with different regular layouts
- Writing code that “thinks in coordinates” rather than in arbitrary node IDs

### 2. Qubits and couplers are first‑class concepts
Inside QArray:

- A qubit is more than just an integer node ID:
    - It has coordinates (row / col)
    - It can have a consistent symbolic name (e.g. "q0", "q1", ...")
- A coupler is more than just an edge (u, v):
    - It has its own identity
    - It explicitly connects exactly two qubits

This makes QArray a good candidate for a project‑wide canonical data structure for the chip topology:

- Easier for teams to agree on a single indexing / naming convention
- Easier to extend to carry physical parameters later (frequency, error rates, gate times, etc.)
- Less boilerplate than building those abstractions on top of a generic graph library from scratch


### 3. Natural support for rectangular and rhombus‑like topologies
Many real‑world devices use variations of 2D layouts:

- Rectangular / square grids
- Brick‑wall (rhombus‑like) connectivity
- Near‑regular layouts with defects (inactive qubits / disabled couplers)

---

## Contributing
Contributions and suggestions are welcome, especially in areas like:

- New topology configurations / helper constructors
- NetworkX and Qiskit adapters
- Better examples, documentation, and tests
- Performance improvements for large‑scale chips

If you are using QArray inside a compiler / routing / simulation project, issues describing your use case are particularly valuable.