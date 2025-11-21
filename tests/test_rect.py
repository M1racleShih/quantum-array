import numpy as np

from qarray import QArray
from tests.qarray_test import QArrayTestCase


class TestRectangleTopology(QArrayTestCase):
    def _gen_array(self, start_from0: bool = False):
        return (
            QArray(self.rows, self.cols, 0, "rect")
            if start_from0
            else QArray(self.rows, self.cols, 1, "rect")
        )

    def test_slice(self):
        # base 1
        q = self._gen_array()
        self.assertEqual(q[10], "q11")
        self.assertEqual(q[1, 1], "q8")
        self.assertEqual(q[2, :].tolist(), [f"q{i}" for i in range(13, 19)])
        self.assertEqual(q[:, 3].tolist(), [f"q{i}" for i in range(4, 40, 6)])
        self.assertEqual(q[-1, :].tolist(), [f"q{i}" for i in range(31, 37)])
        self.assertEqual(
            q[2:5, 2:5].tolist(),
            [["q15", "q16", "q17"], ["q21", "q22", "q23"], ["q27", "q28", "q29"]],
        )

        # base 0
        q = self._gen_array(True)
        self.assertEqual(q[10], "q10")
        self.assertEqual(q[1, 1], "q7")
        self.assertEqual(q[2, :].tolist(), [f"q{i}" for i in range(12, 18)])
        self.assertEqual(q[:, 3].tolist(), [f"q{i}" for i in range(3, 39, 6)])
        self.assertEqual(q[-1, :].tolist(), [f"q{i}" for i in range(30, 36)])
        self.assertEqual(
            q[2:5, 2:5].tolist(),
            [["q14", "q15", "q16"], ["q20", "q21", "q22"], ["q26", "q27", "q28"]],
        )

    def test_qubit_to_idx(self):
        # base 1
        q = self._gen_array()
        for i in range(self.rows * self.cols):
            self.assertEqual(q.qubit_to_idx(f"q{i+1}"), i)

        # base 0
        q = self._gen_array(True)
        for i in range(self.rows * self.cols):
            self.assertEqual(q.qubit_to_idx(f"q{i}"), i)

    def test_qubit_to_rc(self):
        # base 1
        q = self._gen_array()
        self.assertEqual(q.qubit_to_rc("q12"), (1, 5))

    def test_idx_to_qubit(self):
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
        # base 1
        q = self._gen_array()

        # 测试基本转换
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
        with self.assertRaises(Exception):
            q.rc_to_idx(0, -1)
        with self.assertRaises(Exception):
            q.rc_to_idx(0, self.cols)

    def test_idx_to_rc(self):
        # base 1
        q = self._gen_array()

        # 测试基本转换
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

        # 测试边界情况
        with self.assertRaises(Exception):
            q.idx_to_rc(-1)
        with self.assertRaises(Exception):
            q.idx_to_rc(self.rows * self.cols)

    def test_couplers_of(self):
        # base 1
        q = self._gen_array()

        # 测试中心位置的量子比特
        center_couplers = q.couplers_of("q15")
        self.assertEqual(
            sorted(center_couplers), sorted(["c9-15", "c14-15", "c15-16", "c15-21"])
        )

        # 测试角落位置的量子比特
        corner_couplers = q.couplers_of("q1")
        self.assertEqual(sorted(corner_couplers), sorted(["c1-2", "c1-7"]))

        # 测试边缘位置的量子比特
        edge_couplers = q.couplers_of("q4")
        self.assertEqual(sorted(edge_couplers), sorted(["c3-4","c4-5", "c4-10"]))

        # 测试边界情况的量子比特
        bottom_right_couplers = q.couplers_of("q36")
        self.assertEqual(sorted(bottom_right_couplers), sorted(["c30-36", "c35-36"]))

        # base 0
        q0 = self._gen_array(True)

        center_couplers0 = q0.couplers_of("q14")
        self.assertEqual(
            sorted(center_couplers0), sorted(["c8-14", "c13-14", "c14-15", "c14-20"])
        )

        # 测试无效量子比特标签
        with self.assertRaises(Exception):
            q.couplers_of("invalid_label")
        with self.assertRaises(Exception):
            q.couplers_of("q100")

    def test_couplers_all(self):
        # base 1
        q = self._gen_array()
        all_couplers = q.couplers_all()

        # 验证耦合器数量（6x6网格的水平耦合器：6行×5列，垂直耦合器：5行×6列）
        expected_count = 6 * 5 + 5 * 6  # 30 + 30 = 60
        self.assertEqual(len(all_couplers), expected_count)

        # 验证耦合器格式
        for coupler in all_couplers:
            self.assertTrue(coupler.startswith("c"))
            self.assertIn("-", coupler)

        # 验证包含一些特定的耦合器
        self.assertIn("c1-2", all_couplers)
        self.assertIn("c1-7", all_couplers)
        self.assertIn("c35-36", all_couplers)
        self.assertIn("c30-36", all_couplers)

        # base 0
        q0 = self._gen_array(True)
        all_couplers0 = q0.couplers_all()

        # 验证耦合器数量相同
        self.assertEqual(len(all_couplers0), expected_count)

        # 验证包含一些特定的耦合器（base 0）
        self.assertIn("c0-1", all_couplers0)
        self.assertIn("c0-6", all_couplers0)
        self.assertIn("c34-35", all_couplers0)
        self.assertIn("c29-35", all_couplers0)

    def test_couplsers_at(self):
        # base 1
        q = self._gen_array()

        # 测试中心位置
        center_couplers = q.couplers.at(2, 2)
        expected_center = sorted(["c15-16", "c15-21"])  # 右和下的耦合器
        self.assertEqual(sorted(center_couplers), expected_center)

        # 测试左上角位置
        top_left_couplers = q.couplers.at(0, 0)
        expected_top_left = sorted(["c1-2", "c1-7"])  # 右和下的耦合器
        self.assertEqual(sorted(top_left_couplers), expected_top_left)

        # 测试右下角位置
        bottom_right_couplers = q.couplers.at(5, 5)
        expected_bottom_right = []  # 右下角没有向外的耦合器
        self.assertEqual(sorted(bottom_right_couplers), expected_bottom_right)

        # 测试右边缘位置（只有向下的耦合器）
        right_edge_couplers = q.couplers.at(2, 5)
        expected_right_edge = sorted(["c18-24"])  # 只有向下的耦合器
        self.assertEqual(sorted(right_edge_couplers), expected_right_edge)

        # 测试下边缘位置（只有向右的耦合器）
        bottom_edge_couplers = q.couplers.at(5, 2)
        expected_bottom_edge = sorted(["c33-34"])  # 只有向右的耦合器
        self.assertEqual(sorted(bottom_edge_couplers), expected_bottom_edge)

        # base 0
        q0 = self._gen_array(True)

        # 测试相似的位置
        center_couplers0 = q0.couplers.at(2, 2)
        expected_center0 = sorted(["c14-15", "c14-20"])  # 右和下的耦合器
        self.assertEqual(sorted(center_couplers0), expected_center0)

        # 测试边界情况
        with self.assertRaises(Exception):
            q.couplers.at(-1, 0)
        with self.assertRaises(Exception):
            q.couplers.at(6, 0)

    def test_couplsers_get_loc(self):
        # base 1
        q = self._gen_array()

        # 测试水平耦合器的位置
        loc = q.couplers.get_loc("c1-2")
        self.assertEqual(loc, (0, 0, 0))  # 第0行，第0列，通道0（水平）

        loc = q.couplers.get_loc("c1-7")
        self.assertEqual(loc, (0, 0, 1))  # 第0行，第0列，通道1（垂直）

        # 测试垂直耦合器的位置
        loc = q.couplers.get_loc("c2-8")
        self.assertEqual(loc, (0, 1, 1))  # 第0行，第1列，通道1（垂直）

        # 测试中心位置的耦合器
        loc = q.couplers.get_loc("c15-16")
        self.assertEqual(loc, (2, 2, 0))  # 第2行，第2列，通道0（水平）

        loc = q.couplers.get_loc("c15-21")
        self.assertEqual(loc, (2, 2, 1))  # 第2行，第2列，通道1（垂直）

        # 测试无效耦合器标签
        self.assertIsNone(q.couplers.get_loc("c100-101"))
        self.assertIsNone(q.couplers.get_loc("invalid_label"))

        # base 0
        q0 = self._gen_array(True)

        # 测试基座为0时的耦合器位置
        loc0 = q0.couplers.get_loc("c0-1")
        self.assertEqual(loc0, (0, 0, 0))  # 第0行，第0列，通道0（水平）

        loc0 = q0.couplers.get_loc("c0-6")
        self.assertEqual(loc0, (0, 0, 1))  # 第0行，第0列，通道1（垂直）

    def test_couplsers_slice(self):
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
        horizontal_slice = q.couplers[:, :, 0]  # 所有水平耦合器
        self.assertEqual(horizontal_slice.shape, (6, 6))

        vertical_slice = q.couplers[:, :, 1]  # 所有垂直耦合器
        self.assertEqual(vertical_slice.shape, (6, 6))

        # 测试子区域切片
        center_slice = q.couplers[2:4, 2:4, :]
        self.assertEqual(center_slice.shape, (2, 2, 2))

        # 验证切片内容
        # 检查中心区域的水平耦合器
        center_horizontal = q.couplers[2:4, 2:4, 0]
        expected_horizontal = np.array([["c15-16", "c16-17"], ["c21-22", "c22-23"]])
        np.testing.assert_array_equal(center_horizontal, expected_horizontal)

        # 验证边缘情况
        edge_slice = q.couplers[0:1, 0:1, :]
        self.assertEqual(edge_slice.shape, (1, 1, 2))

        # base 0
        q0 = self._gen_array(True)
        center_slice0 = q0.couplers[2:4, 2:4, 0]
        expected_horizontal0 = np.array([["c14-15", "c15-16"], ["c20-21", "c21-22"]])
        np.testing.assert_array_equal(center_slice0, expected_horizontal0)
