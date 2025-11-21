import numpy as np
from qarray import QArray
from tests.qarray_test import QArrayTestCase


class TestRhombusTopology(QArrayTestCase):
    def _gen_array(self, start_from0: bool = False):
        return (
            QArray(self.rows, self.cols, 0, "rhombus")
            if start_from0
            else QArray(self.rows, self.cols, 1, "rhombus")
        )

    def test_slice(self):
        """测试菱形拓扑的切片功能"""
        # base 1
        q = self._gen_array()
        self.assertEqual(q[10], "q11")
        self.assertEqual(q[1, 1], "q8")
        
        # 测试行切片
        self.assertEqual(q[2, :].tolist(), [f"q{i}" for i in range(13, 19)])
        
        # 测试列切片
        self.assertEqual(q[:, 3].tolist(), [f"q{i}" for i in range(4, 40, 6)])
        
        # 测试负数索引
        self.assertEqual(q[-1, :].tolist(), [f"q{i}" for i in range(31, 37)])
        
        # 测试子数组切片
        subarray = q[2:5, 2:5].tolist()
        expected = [["q15", "q16", "q17"], ["q21", "q22", "q23"], ["q27", "q28", "q29"]]
        self.assertEqual(subarray, expected)

        # base 0
        q = self._gen_array(True)
        self.assertEqual(q[10], "q10")
        self.assertEqual(q[1, 1], "q7")
        self.assertEqual(q[2, :].tolist(), [f"q{i}" for i in range(12, 18)])
        self.assertEqual(q[:, 3].tolist(), [f"q{i}" for i in range(3, 39, 6)])
        self.assertEqual(q[-1, :].tolist(), [f"q{i}" for i in range(30, 36)])
        
        subarray0 = q[2:5, 2:5].tolist()
        expected0 = [["q14", "q15", "q16"], ["q20", "q21", "q22"], ["q26", "q27", "q28"]]
        self.assertEqual(subarray0, expected0)

    def test_qubit_to_idx(self):
        """测试量子比特标签到索引的转换"""
        # base 1
        q = self._gen_array()
        for i in range(self.rows * self.cols):
            self.assertEqual(q.qubit_to_idx(f"q{i+1}"), i)

        # base 0
        q = self._gen_array(True)
        for i in range(self.rows * self.cols):
            self.assertEqual(q.qubit_to_idx(f"q{i}"), i)
        
        # 测试无效标签
        with self.assertRaises(Exception):
            q.qubit_to_idx("invalid")
        with self.assertRaises(Exception):
            q.qubit_to_idx("q100")

    def test_qubit_to_rc(self):
        """测试量子比特标签到行列坐标的转换"""
        # base 1
        q = self._gen_array()
        self.assertEqual(q.qubit_to_rc("q12"), (1, 5))
        self.assertEqual(q.qubit_to_rc("q1"), (0, 0))
        self.assertEqual(q.qubit_to_rc("q36"), (5, 5))

    def test_idx_to_qubit(self):
        """测试索引到量子比特标签的转换"""
        # base 1
        q = self._gen_array()
        for i in range(self.rows * self.cols):
            self.assertEqual(q.idx_to_qubit(i), f"q{i+1}")
        
        # base 0
        q = self._gen_array(True)
        for i in range(self.rows * self.cols):
            self.assertEqual(q.idx_to_qubit(i), f"q{i}")
        
        # 测试边界情况
        with self.assertRaises(Exception):
            q.idx_to_qubit(-1)
        with self.assertRaises(Exception):
            q.idx_to_qubit(self.rows * self.cols)

    def test_rc_to_idx(self):
        """测试行列坐标到索引的转换"""
        # base 1
        q = self._gen_array()
        for r in range(self.rows):
            for c in range(self.cols):
                expected_idx = r * self.cols + c
                self.assertEqual(q.rc_to_idx(r, c), expected_idx)
        
        # base 0
        q = self._gen_array(True)
        for r in range(self.rows):
            for c in range(self.cols):
                expected_idx = r * self.cols + c
                self.assertEqual(q.rc_to_idx(r, c), expected_idx)
        
        # 测试边界情况
        with self.assertRaises(Exception):
            q.rc_to_idx(-1, 0)
        with self.assertRaises(Exception):
            q.rc_to_idx(self.rows, 0)

    def test_idx_to_rc(self):
        """测试索引到行列坐标的转换"""
        # base 1
        q = self._gen_array()
        for i in range(self.rows * self.cols):
            expected_r = i // self.cols
            expected_c = i % self.cols
            self.assertEqual(q.idx_to_rc(i), (expected_r, expected_c))
        
        # base 0
        q = self._gen_array(True)
        for i in range(self.rows * self.cols):
            expected_r = i // self.cols
            expected_c = i % self.cols
            self.assertEqual(q.idx_to_rc(i), (expected_r, expected_c))

    def test_couplers_of_rhombus_specific(self):
        """测试菱形拓扑特有的连接模式"""
        # base 1
        q = self._gen_array()
        
        # 测试偶数行的量子比特（连接到左下方）
        even_row_couplers = q.couplers_of("q15")  # 第2行是偶数行（0-based）
        expected_even = sorted(["c8-15", "c9-15", "c15-20", "c15-21"])
        self.assertEqual(sorted(even_row_couplers), expected_even)
        
        # 测试奇数行的量子比特（连接到右下方）  
        odd_row_couplers = q.couplers_of("q22")  # 第3行是奇数行（0-based）
        expected_odd = sorted(["c16-22", "c17-22", "c22-28", "c22-29"])
        self.assertEqual(sorted(odd_row_couplers), expected_odd)
        
        # 测试角落量子比特
        corner_couplers = q.couplers_of("q1")
        expected_corner = sorted(["c1-7"])
        self.assertEqual(sorted(corner_couplers), expected_corner)
        
        # base 0
        q0 = self._gen_array(True)
        even_row_couplers0 = q0.couplers_of("q14")
        expected_even0 = sorted(["c7-14", "c8-14", "c14-19", "c14-20"])
        self.assertEqual(sorted(even_row_couplers0), expected_even0)

    def test_couplers_all_rhombus(self):
        """测试获取所有菱形拓扑耦合器"""
        # base 1
        q = self._gen_array()
        all_couplers = q.couplers_all()
        
        # 菱形拓扑的耦合器数量（垂直耦合器 + 对角线耦合器）
        # 垂直耦合器：5行 × 6列 = 30
        # 对角线耦合器：5行 × 5列 = 25（偶数行：左侧对角线，奇数行：右侧对角线）
        expected_count = 30 + 25  # 55个耦合器
        self.assertEqual(len(all_couplers), expected_count)
        
        # 验证包含特定的菱形拓扑耦合器
        self.assertIn("c1-7", all_couplers)  # 垂直耦合器
        self.assertIn("c7-14", all_couplers)  # 对角线耦合器（偶数行）
        self.assertIn("c29-34", all_couplers)  # 对角线耦合器（奇数行）
        
        # base 0
        q0 = self._gen_array(True)
        all_couplers0 = q0.couplers_all()
        self.assertEqual(len(all_couplers0), expected_count)

    def test_couplers_at_rhombus(self):
        """测试在特定位置获取菱形拓扑耦合器"""
        # base 1
        q = self._gen_array()
        
        # 测试偶数行位置（第2行是偶数行，0-based）
        even_row_couplers = q.couplers.at(2, 2)
        expected_even = sorted(["c15-20", "c15-21"])  # 左下和垂直下
        self.assertEqual(sorted(even_row_couplers), expected_even)
        
        # 测试奇数行位置（第3行是奇数行，0-based）
        odd_row_couplers = q.couplers.at(3, 2)
        expected_odd = sorted(["c21-27", "c21-28"])  # 右下和垂直下
        self.assertEqual(sorted(odd_row_couplers), expected_odd)
        
        # 测试边界情况
        with self.assertRaises(Exception):
            q.couplers.at(-1, 0)

    def test_couplers_get_loc_rhombus(self):
        """测试获取菱形拓扑耦合器的位置"""
        # base 1
        q = self._gen_array()
        
        # 测试垂直耦合器
        loc_vertical = q.couplers.get_loc("c1-7")
        self.assertEqual(loc_vertical, (0, 0, 0))  # 第0行，第0列，通道0（垂直）
        
        # 测试对角线耦合器（偶数行）
        loc_diagonal_even = q.couplers.get_loc("c3-8")
        self.assertEqual(loc_diagonal_even, (0, 2, 1))  # 第0行，第2列，通道1（对角线）
        
        # 测试对角线耦合器（奇数行）
        loc_diagonal_odd = q.couplers.get_loc("c21-28")
        self.assertEqual(loc_diagonal_odd, (3, 2, 1))  # 第1行，第0列，通道1（对角线）

    def test_couplers_slice_rhombus(self):
        """测试菱形拓扑耦合器的切片功能"""
        # base 1
        q = self._gen_array()

        # 测试单行切片
        row_slice = q.couplers[2, :, :]
        # 第2行应该有6列，每列有2个通道
        self.assertEqual(row_slice.shape, (6, 2))

        # 测试单列切片
        col_slice = q.couplers[:, 3, :]
        # 第3列应该有6行，每行有2个通道
        self.assertEqual(col_slice.shape, (6, 2))

        # 测试特定通道切片
        horizontal_slice = q.couplers[:, :, 0]  # 所有斜耦合器
        self.assertEqual(horizontal_slice.shape, (6, 6))

        vertical_slice = q.couplers[:, :, 1]  # 所有垂直耦合器
        self.assertEqual(vertical_slice.shape, (6, 6))

        # 测试子区域切片
        center_slice = q.couplers[2:4, 2:4, :]
        self.assertEqual(center_slice.shape, (2, 2, 2))

        # 验证切片内容
        # 检查中心区域的水平耦合器
        center_horizontal = q.couplers[2:4, 2:4, 0]
        expected_horizontal = np.array([["c15-21", "c16-22"], ["c21-27", "c22-28"]])
        np.testing.assert_array_equal(center_horizontal, expected_horizontal)

        # 验证边缘情况
        edge_slice = q.couplers[0:1, 0:1, :]
        self.assertEqual(edge_slice.shape, (1, 1, 2))

        # base 0
        q0 = self._gen_array(True)
        center_slice0 = q0.couplers[2:4, 2:4, 0]
        expected_horizontal0 = np.array([["c14-20", "c15-21"], ["c20-26", "c21-27"]])
        np.testing.assert_array_equal(center_slice0, expected_horizontal0)