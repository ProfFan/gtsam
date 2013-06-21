/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file RangeISAMExample_plaza1.cpp
 * @brief A 2D Range SLAM example
 * @date June 20, 2013
 * @author FRank Dellaert
 */

// Both relative poses and recovered trajectory poses will be stored as Pose2 objects
#include <gtsam/geometry/Pose2.h>

// Each variable in the system (poses and landmarks) must be identified with a unique key.
// We can either use simple integer keys (1, 2, 3, ...) or symbols (X1, X2, L1).
// Here we will use Symbols
#include <gtsam/nonlinear/Symbol.h>

// We want to use iSAM2 to solve the range-SLAM problem incrementally
#include <gtsam/nonlinear/ISAM2.h>

// iSAM2 requires as input a set set of new factors to be added stored in a factor graph,
// and initial guesses for any new variables used in the added factors
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

// We will use a non-liear solver to batch-inituialize from the first 150 frames
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics SLAM problems.
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/RangeFactor.h>

// Standard headers, added last, so we know headers above work on their own
#include <boost/foreach.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace gtsam;

// data available at http://www.frc.ri.cmu.edu/projects/emergencyresponse/RangeData/
// Datafile format (from http://www.frc.ri.cmu.edu/projects/emergencyresponse/RangeData/log.html)

// load the odometry
// DR: Odometry Input (delta distance traveled and delta heading change)
//    Time (sec)  Delta Dist. Trav. (m) Delta Heading (rad)
typedef pair<double, Pose2> TimedOdometry;
list<TimedOdometry> readOdometry() {
  list<TimedOdometry> odometryList;
  ifstream is("../../examples/Data/Plaza2_DR.txt");
  if (!is)
    throw runtime_error("../../examples/Data/Plaza2_DR.txt file not found");

  while (is) {
    double t, distance_traveled, delta_heading;
    is >> t >> distance_traveled >> delta_heading;
    odometryList.push_back(
        TimedOdometry(t, Pose2(distance_traveled, 0, delta_heading)));
  }
  is.clear(); /* clears the end-of-file and error flags */
  return odometryList;
}

// load the ranges from TD
//    Time (sec)  Sender / Antenna ID Receiver Node ID  Range (m)
typedef boost::tuple<double, size_t, double> RangeTriple;
vector<RangeTriple> readTriples() {
  vector<RangeTriple> triples;
  ifstream is("../../examples/Data/Plaza2_TD.txt");
  if (!is)
    throw runtime_error("../../examples/Data/Plaza2_TD.txt file not found");

  while (is) {
    double t, sender, receiver, range;
    is >> t >> sender >> receiver >> range;
    triples.push_back(RangeTriple(t, receiver, range));
  }
  is.clear(); /* clears the end-of-file and error flags */
  return triples;
}

// main
int main(int argc, char** argv) {

  // load Plaza2 data
  list<TimedOdometry> odometry = readOdometry();
//  size_t M = odometry.size();

  vector<RangeTriple> triples = readTriples();
  size_t K = triples.size();

  // parameters
  size_t minK = 150; // minimum number of range measurements to process initially
  size_t incK = 5; // minimum number of range measurements to process after
  double sigmaR = 100; // range standard deviation
  bool batchInitialization = true;

  // Set Noise parameters
  const noiseModel::Robust::shared_ptr rangeNoiseModel =
      noiseModel::Robust::Create(noiseModel::mEstimator::Tukey::Create(15),
          noiseModel::Isotropic::Sigma(1, sigmaR));

  // Initialize iSAM
  ISAM2 isam;

  // Add prior on first pose
  Pose2 pose0 = Pose2(-34.2086489999201, 45.3007639991120,
      M_PI - 2.02108900000000);
  NonlinearFactorGraph newFactors;
  newFactors.add(
      PriorFactor<Pose2>(0, pose0,
          noiseModel::Diagonal::Sigmas(Vector_(3, 1.0, 1.0, M_PI))));
  Values initial;
  initial.insert(0, pose0);

  //  initialize points drawn from sigma=1 Gaussian in matlab version
  initial.insert(symbol('L', 1), Point2(3.5784, 2.76944));
  initial.insert(symbol('L', 6), Point2(-1.34989, 3.03492));
  initial.insert(symbol('L', 0), Point2(0.725404, -0.0630549));
  initial.insert(symbol('L', 5), Point2(0.714743, -0.204966));

  // Loop over odometry
  gttic_(iSAM);
  size_t i = 1; // step counter
  size_t k = 0; // range measurement counter
  bool update = false;
  Pose2 lastPose = pose0;
  size_t countK = 0;
  BOOST_FOREACH(const TimedOdometry& timedOdometry, odometry) {
    double t;
    Pose2 odometry;
    boost::tie(t, odometry) = timedOdometry;

    // add odometry factor
    newFactors.add(
        BetweenFactor<Pose2>(i - 1, i, odometry,
            noiseModel::Diagonal::Sigmas(Vector_(3, 0.05, 0.01, 0.2))));

    // predict pose and add as initial estimate
    Pose2 predictedPose = lastPose.compose(odometry);
    lastPose = predictedPose;
    initial.insert(i, predictedPose);

    // Check if there are range factors to be added
    while (k < K && t >= boost::get<0>(triples[k])) {
      size_t j = boost::get<1>(triples[k]);
      double range = boost::get<2>(triples[k]);
      newFactors.add(
          RangeFactor<Pose2, Point2>(i, symbol('L', j), range,
              rangeNoiseModel));
      k = k + 1;
      countK = countK + 1;
      update = true;
    }

    // Check whether to update iSAM 2
    if (update && (k > minK) && (countK > incK)) {
      if (batchInitialization) { // Do a full optimize for first minK ranges
        gttic_(batchInitialization);
        LevenbergMarquardtOptimizer batchOptimizer(newFactors, initial);
        initial = batchOptimizer.optimize();
        gttoc_(batchInitialization);
        batchInitialization = false; // only once
      }
      gttic_(update);
      isam.update(newFactors, initial);
      gttoc_(update);
      gttic_(calculateEstimate);
      Values result = isam.calculateEstimate();
      gttoc_(calculateEstimate);
      lastPose = result.at<Pose2>(i);
      newFactors = NonlinearFactorGraph();
      initial = Values();
      countK = 0;
    }
    i += 1;
  } // odometry loop
  gttoc_(iSAM);

  // Print timings
  tictoc_print_();

  exit(0);
}

