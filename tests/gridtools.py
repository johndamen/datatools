import unittest
import datatools
import numpy as np


class ArrayTestCase(unittest.TestCase):

    def assertEqualArrays(self, A, B, e=0):
        self.assertEqual(A.shape, B.shape, 'arrays of different shape')
        self.assertEqual(A.dtype, B.dtype, 'arrays of different type')
        self.assertTrue((np.absolute(A - B) <= e).all(), 'arrays have different values')


class TestGridTools(ArrayTestCase):

    def test_aggregate_inputs(self):
        with self.assertRaises(NotImplementedError):
            datatools.aggregate(np.random.rand(100), 3)

        with self.assertRaises(NotImplementedError):
            datatools.aggregate(np.random.rand(30, 30, 30), 3)

        self.assertEqualArrays(
            datatools.aggregate(np.ones((30, 30)), 3),
            np.ones((10, 10, 9)))

    def test_aggregate_simple(self):
        A = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]])

        B = np.array([
            [[0, 0, 0, 0],
             [1, 1, 1, 1]],
            [[2, 2, 2, 2],
             [3, 3, 3, 3]]
        ])

        self.assertEqualArrays(datatools.aggregate(A, 2), B)

    def test_aggregate_no_fit(self):
        A = np.array([
            [0, 0, 1, 1, 9],
            [0, 0, 1, 1, 9],
            [2, 2, 3, 3, 9],
            [2, 2, 3, 3, 9],
            [9, 9, 9, 9, 9]])

        B = np.array([
            [[0, 0, 0, 0],
             [1, 1, 1, 1]],
            [[2, 2, 2, 2],
             [3, 3, 3, 3]]
        ])

        self.assertEqualArrays(datatools.aggregate(A, 2), B)

    def test_aggregate_asym(self):
        A = np.array([
            [0, 0, 1, 1, 2, 2, 9],
            [0, 0, 1, 1, 2, 2, 9],
            [3, 3, 4, 4, 5, 5, 9],
            [3, 3, 4, 4, 5, 5, 9],
            [9, 9, 9, 9, 9, 9, 9]
        ])

        B = np.array([
            [[0, 0, 0, 0],
             [1, 1, 1, 1],
             [2, 2, 2, 2]],
            [[3, 3, 3, 3],
             [4, 4, 4, 4],
             [5, 5, 5, 5]],
        ])

        self.assertEqualArrays(datatools.aggregate(A, 2), B)


class TestBlockMean(ArrayTestCase):

    def test_simple(self):
        A = np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16]
        ])

        B = np.array([
            [ 3.5,  5.5],
            [11.5, 13.5]
        ])

        self.assertEqualArrays(datatools.block_mean(A, 2), B)


class TestBlockFunc(ArrayTestCase):

    def test_array_func(self):
        A = np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16]
        ])

        B = np.array([
            [ 6,  8],
            [14, 16]
        ])

        self.assertEqualArrays(datatools.block_func(A, 2, np.amax), B)

    def test_python_func(self):
        A = np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16]
        ])

        B = np.array([
            [ 6,  8],
            [14, 16]
        ])

        self.assertEqualArrays(datatools.block_func(A, 2, max), B)

        def foo(a): return max(a)
        self.assertEqualArrays(datatools.block_func(A, 2, foo), B)

    def test_lambda_func(self):
        A = np.array([
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16]
        ])

        B = np.array([
            [ 5,  7],
            [13, 15]
        ])

        self.assertEqualArrays(datatools.block_func(A, 2, lambda x: x[2]), B)
