//
// Created by yzc on 2019/11/19.
//
#include <string>
#include <vector>

#include "base/ba_graph.h"
#include "optim/ceres_cost_function.h"
#include "run_ba_solver.h"
#include "utility/global.h"

using namespace ceres;

void AddParamblock(Problem &problem, Solver::Options &solver_options, coli::BAGraph &ba_graph) {
  constexpr int kLandmarkGroup = 0, kOtherGroup = 1;
  solver_options.linear_solver_ordering = std::make_shared<ceres::ParameterBlockOrdering>();

  for (auto &cam : ba_graph.cam_vec) {
    if (ba_graph.auto_diff) {
      problem.AddParameterBlock(cam.data(), 4, new ceres::QuaternionParameterization);
    } else {
      problem.AddParameterBlock(cam.data(), 4, new EigenQuatParam);
    }
    problem.AddParameterBlock(cam.data() + 4, 3);

    solver_options.linear_solver_ordering->AddElementToGroup(cam.data(), kOtherGroup);
    solver_options.linear_solver_ordering->AddElementToGroup(cam.data() + 4, kOtherGroup);
  }

  for (auto &pt : ba_graph.point_vec) {
    problem.AddParameterBlock(pt.data(), 3);
    solver_options.linear_solver_ordering->AddElementToGroup(pt.data(), kLandmarkGroup);
  }

  // solver_options.linear_solver_ordering = nullptr;
  // problem.SetParameterBlockConstant(ba_graph.cam_vec[0].data());
  // problem.SetParameterBlockConstant(ba_graph.cam_vec[0].data() + 4);
  // problem.SetParameterBlockConstant(ba_graph.cam_vec[1].data());
  // problem.SetParameterBlockConstant(ba_graph.cam_vec[1].data() + 4);
  // problem.SetParameterBlockConstant(ba_graph.point_vec[0].data());
}

void AddResidualblock(Problem &problem, coli::BAGraph &ba_graph) {
  auto &obs_vec = ba_graph.obs_vec;
  auto &cam_vec = ba_graph.cam_vec;
  auto &point_vec = ba_graph.point_vec;

  ceres::LossFunction *loss_function = nullptr;
  ceres::CostFunction *cost_function = nullptr;
  for (auto &obs : obs_vec) {
    auto &cam = cam_vec[obs.camera_id];
    double *qvec_ptr = cam.data(), *tvec_ptr = cam.data() + 4;

    double *point3d_ptr = point_vec[obs.point_id].data();
    if (ba_graph.auto_diff) {
      cost_function = ProjectionCostAuto::Create(obs.mea, ba_graph.use_ntwc);
    } else {
      CHECK(ba_graph.use_ntwc);
      cost_function = new ProjectionCost(obs.mea);
    }
    problem.AddResidualBlock(cost_function, loss_function, qvec_ptr, tvec_ptr, point3d_ptr);
  }
}

void SolveProblem(Problem &problem, Solver::Options &solver_options, coli::BAGraph &ba_graph) {
  solver_options.minimizer_progress_to_stdout = true;

  solver_options.num_threads = 1;
  solver_options.jacobi_scaling = false;
  solver_options.max_lm_diagonal = solver_options.min_lm_diagonal = 1;
  solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  solver_options.use_explicit_schur_complement = true;
  solver_options.max_linear_solver_iterations = 100;
  solver_options.update_state_every_iteration = true;

  std::map<int, ceres::LinearSolverType> type_map = {
      {0, ceres::DENSE_SCHUR}, {1, ceres::SPARSE_SCHUR}, {2, ceres::ITERATIVE_SCHUR}};
  solver_options.linear_solver_type = type_map.at(ba_graph.solve_type);
  solver_options.max_num_iterations = _max_iter_;

  Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  // std::cout << summary.FullReport() << "\n";
}

void RunCeres(coli::BAGraph &ba_graph) {
  Problem problem;
  Solver::Options solver_options;
  AddParamblock(problem, solver_options, ba_graph);
  AddResidualblock(problem, ba_graph);
  SolveProblem(problem, solver_options, ba_graph);
}
