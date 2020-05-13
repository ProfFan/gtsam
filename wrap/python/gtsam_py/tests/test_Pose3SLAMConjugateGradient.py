"""
GTSAM Copyright 2010-2019, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information

PoseSLAM unit tests.
Author: Frank Dellaert & Duy Nguyen Ta (Python)
"""
import unittest
from math import pi

import numpy as np

import gtsam
from gtsam.utils.test_case import GtsamTestCase
from gtsam.utils.circlePose3 import *


class TestPose3SLAMConjugateGradient(GtsamTestCase):

    def test_Pose3SLAMConjugateGradient(self):
        # Create a hexagon of poses
        hexagon = circlePose3(6, 1.0)
        p0 = hexagon.atPose3(0)
        p1 = hexagon.atPose3(1)
        print(hexagon)
        # create a Pose graph with one equality constraint and one measurement
        fg = gtsam.NonlinearFactorGraph()
        covariance = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1, 1, 1, 1, 1, 1]))
        fg.add(gtsam.PriorFactorPose3(0, p0, covariance))
        delta = p0.between(p1)

        fg.add(gtsam.BetweenFactorPose3(0, 1, delta, covariance))
        fg.add(gtsam.BetweenFactorPose3(1, 2, delta, covariance))
        fg.add(gtsam.BetweenFactorPose3(2, 3, delta, covariance))
        fg.add(gtsam.BetweenFactorPose3(3, 4, delta, covariance))
        fg.add(gtsam.BetweenFactorPose3(4, 5, delta, covariance))
        fg.add(gtsam.BetweenFactorPose3(5, 0, delta, covariance))

        # Create initial config
        initial = gtsam.Values()
        s = 0.10
        initial.insert(0, p0)
        rand = s * np.array([  0.10451689099851265,  -0.06877252474465692,   0.04898364467823929,
                           -0.17876020673335719, -0.006401511208322003, -0.011191052845134497])
        initial.insert(1, hexagon.atPose3(1).retract(rand))
        initial.insert(2, hexagon.atPose3(2).retract(rand))
        initial.insert(3, hexagon.atPose3(3).retract(rand))
        initial.insert(4, hexagon.atPose3(4).retract(rand))
        initial.insert(5, hexagon.atPose3(5).retract(rand))

        linearized = fg.linearize(initial)
        print(linearized)
        print("Initial error = ", fg.error(initial))
        # optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(fg, initial)
        result = optimizer.optimizeSafely()

        pose_1 = result.atPose3(1)
        self.gtsamAssertEquals(pose_1, p1, 1e-1)

if __name__ == "__main__":
    unittest.main()
