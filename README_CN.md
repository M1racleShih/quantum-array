# Quantum Array - 量子芯片布局管理器

**⚠️ 项目状态：开发中**  
该项目目前处于积极开发阶段。未来版本中API可能会发生重大变化。

## 概述

Quantum Array 是一个用于管理量子芯片二维逻辑布局的Python库。它为各种拓扑配置下的量子比特和耦合器提供了高效的矩阵表示和操作接口。

## 功能特性

- **多拓扑支持**：矩形、菱形和砖块6种拓扑结构
- **灵活索引**：支持0-based和1-based索引方式
- **高效矩阵运算**：基于NumPy的高性能计算
- **完整坐标转换**：线性索引 ↔ 行列坐标 ↔ 量子比特标签
- **耦合器管理**：自动生成耦合器标签和邻接关系查询

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/quantum-array.git
cd quantum-array

# 安装依赖
pip install numpy
```

## 快速开始

```python
from qarray import QArray

# 创建4x6矩形量子阵列
qarray = QArray(rows=4, cols=6, index_base=1, topology="rect")

# 使用多种索引方式访问量子比特
print(qarray[0, 0])  # 输出: 'q1'
print(qarray[1, :])  # 输出: 量子比特标签数组

# 获取特定量子比特的所有耦合器
couplers = qarray.couplers_of("q8")
print(couplers)  # 输出: ['c7-8', 'c8-9', 'c8-14', 'c2-8']
```

## 拓扑示例

### 矩形拓扑
```python
rect_array = QArray(4, 6, topology="rect")
# 标准的4邻接网格连接
```

### 菱形拓扑
```python
rhombus_array = QArray(4, 6, topology="rhombus")
# 具有交替模式的斜向连接
```

## 开发路线图

- [ ] 实现Brick6拓扑支持
- [ ] 添加可视化功能
- [ ] 增强性能优化
- [ ] 扩展测试覆盖
- [ ] 添加文档和示例

## 贡献指南

欢迎贡献！请注意项目处于早期开发阶段，API可能会有变化。

## 许可证

本项目采用Apache License 2.0许可证。