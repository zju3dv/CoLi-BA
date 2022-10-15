//
// Created by yzc on 2019/11/19.
//
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/se3quat.h>

#include <string>
#include <vector>

#include "base/ba_graph.h"
#include "optim/ceres_cost_function.h"
#include "run_ba_solver.h"
#include "utility/global.h"

// using SlamLinearSolver = g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>;
using SlamLinearSolver = g2o::LinearSolverPCG<g2o::BlockSolver_6_3::PoseMatrixType>;

void RunG2O(coli::BAGraph& ba_graph) {
  // set optimizer
  g2o::SparseOptimizer optimizer;
  // std::unique_ptr<SlamLinearSolver> linearSolver = g2o::make_unique<SlamLinearSolver>();
  // linearSolver->setMaxIterations(100);
  // // linearSolver->setBlockOrdering(false);
  // std::unique_ptr<g2o::BlockSolver_6_3> blockSolver =
  // g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)); g2o::OptimizationAlgorithmLevenberg* algorithm =
  // new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

  g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(g2o::make_unique<SlamLinearSolver>()));
  optimizer.setAlgorithm(algorithm);

  for (int i = 0; i < ba_graph.cam_vec.size(); ++i) {
    Eigen::Vector<double, 7>& cam = ba_graph.cam_vec[i];
    Eigen::Quaternion qcw(cam.data());
    Eigen::Vector3d twc(-cam(4), -cam(5), -cam(6));
    Eigen::Vector3d tcw = -(qcw * twc);
    g2o::SE3Quat pose(qcw, tcw);

    g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(i);
    v_se3->setEstimate(pose);
    if (i == 0) v_se3->setFixed(true);
    optimizer.addVertex(v_se3);
  }

  const int point_begin_id = 10000;
  assert(point_begin_id > ba_graph.cam_vec.size());
  for (int i = 0; i < ba_graph.point_vec.size(); ++i) {
    Eigen::Vector3d& pt = ba_graph.point_vec[i];
    g2o::VertexPointXYZ* v_p = new g2o::VertexPointXYZ();
    v_p->setId(point_begin_id + i);
    v_p->setMarginalized(true);
    v_p->setEstimate(pt);
    optimizer.addVertex(v_p);
  }

  for (auto& obs : ba_graph.obs_vec) {
    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(point_begin_id + obs.point_id)));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(obs.camera_id)));
    e->setMeasurement(obs.mea);
    e->information() = Eigen::Matrix2d::Identity();
    e->fx = 1.0;
    e->fy = 1.0;
    e->cx = 0;
    e->cy = 0;
    optimizer.addEdge(e);
  }

  optimizer.initializeOptimization();
  optimizer.optimize(_max_iter_);

  for (int i = 0; i < ba_graph.cam_vec.size(); ++i) {
    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    Eigen::Quaterniond q = SE3quat.rotation();
    Eigen::Vector3d trans = q.inverse() * SE3quat.translation();
    ba_graph.cam_vec[i] << q.coeffs(), trans;
  }
  for (int i = 0; i < ba_graph.point_vec.size(); ++i) {
    g2o::VertexPointXYZ* vPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_begin_id + i));
    ba_graph.point_vec[i] = vPoint->estimate();
  }
  optimizer.clear();
}
