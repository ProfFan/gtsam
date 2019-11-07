"""
GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

Unit tests for IMU testing scenarios.
Author: Frank Dellaert & Duy Nguyen Ta (Python)
"""
# pylint: disable=invalid-name, no-name-in-module

from __future__ import print_function

import unittest

import numpy as np
from gtsam_py import gtsam
from gtsam_py.gtsam import (DoglegOptimizer, DoglegParams, GaussNewtonOptimizer,
                   GaussNewtonParams, LevenbergMarquardtOptimizer,
                   LevenbergMarquardtParams, NonlinearFactorGraph, Ordering, PriorFactorPoint2, Values)
from utils.test_case import GtsamTestCase

KEY1 = 1
KEY2 = 2


class TestScenario(GtsamTestCase):
    def test_optimize(self):
        """Do trivial test with three optimizer variants."""
        fg = NonlinearFactorGraph()
        model = gtsam.noiseModel.Unit.Create(2)
        fg.add(PriorFactorPoint2(KEY1, np.array([0, 0]), model))

        # test error at minimum
        xstar = np.array([0, 0])
        optimal_values = Values()
        optimal_values.insert(KEY1, xstar)
        self.assertEqual(0.0, fg.error(optimal_values), 0.0)

        # test error at initial = [(1-cos(3))^2 + (sin(3))^2]*50 =
        x0 = np.array([3, 3])
        initial_values = Values()
        initial_values.insert(KEY1, x0)
        self.assertEqual(9.0, fg.error(initial_values), 1e-3)

        # optimize parameters
        ordering = Ordering()
        ordering.push_back(KEY1)

        # Gauss-Newton
        gnParams = GaussNewtonParams()
        gnParams.setOrdering(ordering)
        actual1 = GaussNewtonOptimizer(fg, initial_values, gnParams).optimize()
        self.assertAlmostEqual(0, fg.error(actual1))

        # Levenberg-Marquardt
        lmParams = LevenbergMarquardtParams.CeresDefaults()
        lmParams.setOrdering(ordering)
        actual2 = LevenbergMarquardtOptimizer(
            fg, initial_values, lmParams).optimize()
        self.assertAlmostEqual(0, fg.error(actual2))

        # Dogleg
        dlParams = DoglegParams()
        dlParams.setOrdering(ordering)
        actual3 = DoglegOptimizer(fg, initial_values, dlParams).optimize()
        self.assertAlmostEqual(0, fg.error(actual3))


if __name__ == "__main__":
    unittest.main()
