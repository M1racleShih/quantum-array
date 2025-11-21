# Quantum Array

[English](#english) | [中文](README_CN.md)

## English

### Quantum Array - Quantum Chip Layout Manager

**⚠️ Project Status: Under Development**  
This project is currently in active development. The API may undergo significant changes in future versions.

### Overview

Quantum Array is a Python library designed for managing 2D logical layouts of quantum chips. It provides efficient matrix representations and operations for quantum bits (qubits) and couplers in various topological configurations.

### Features

- **Multiple Topology Support**: Rectangular, Rhombus, and Brick6 topologies
- **Flexible Indexing**: 0-based and 1-based indexing options
- **Efficient Matrix Operations**: Built on NumPy for high-performance computations
- **Comprehensive Coordinate Conversion**: Linear indices ↔ Row/Column coordinates ↔ Qubit labels
- **Coupler Management**: Automatic coupler label generation and adjacency queries

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/quantum-array.git
cd quantum-array

# Install dependencies
pip install numpy
```

### Quick Start

```python
from qarray import QArray

# Create a 4x6 rectangular quantum array
qarray = QArray(rows=4, cols=6, index_base=1, topology="rect")

# Access qubits using various indexing methods
print(qarray[0, 0])  # Output: 'q1'
print(qarray[1, :])  # Output: array of qubit labels

# Get all couplers connected to a specific qubit
couplers = qarray.couplers_of("q8")
print(couplers)  # Output: ['c7-8', 'c8-9', 'c8-14', 'c2-8']
```

### Topology Examples

#### Rectangular Topology
```python
rect_array = QArray(4, 6, topology="rect")
# Standard 4-neighbor grid connectivity
```

#### Rhombus Topology
```python
rhombus_array = QArray(4, 6, topology="rhombus")
# Slanted connections with alternating patterns
```

### Development Roadmap

- [ ] Implement Brick6 topology support
- [ ] Add visualization capabilities
- [ ] Enhance performance optimizations
- [ ] Expand test coverage
- [ ] Add documentation and examples

### Contributing

We welcome contributions! Please note that the project is in early development, so expect API changes.

### License

This project is licensed under the Apache License 2.0.