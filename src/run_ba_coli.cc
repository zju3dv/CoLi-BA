//
// Created by SENSETIME\yezhichao1 on 2020/4/19.
//
#include "run_ba_solver.h"

#include <string>
#include <vector>

#include "base/ba_graph.h"
#include "optim/ceres_cost_function.h"
#include "optim/lm_minimizer.h"
#include "optim/local_parameterization.h"
#include "utility/global.h"

using namespace coli;

void AddParamblock(BAGraph &ba_graph, std::vector<ParamBlock<default_float>> &param_blocks) {
  const int num_param = 2*ba_graph.cam_vec.size()+ba_graph.point_vec.size();
  param_blocks.reserve(num_param);

  for (auto &cam : ba_graph.cam_vec) {
    param_blocks.emplace_back(ParamBlock<default_float>(cam.data(), 4, new EQuatParam));
    param_blocks.emplace_back(ParamBlock<default_float>(cam.data() + 4, 3));
  }
  for (auto &pt : ba_graph.point_vec) {
    param_blocks.emplace_back(ParamBlock<default_float>(pt.data(), 3));
  }
}

void AddResidualblock(BAGraph &ba_graph, std::vector<ParamBlock<default_float>> &param_blocks,
                      std::vector<ResidualBlock<default_float>> &residual_blocks) { 
  const int num_residual =ba_graph.obs_vec.size();
  residual_blocks.resize(num_residual);

  int count = 0;
  const int num_cam = ba_graph.cam_vec.size();
  for (auto &obs : ba_graph.obs_vec) {
    auto &rb = residual_blocks[count++];
    rb.factor = std::make_unique<ProjectionFactor<default_float>>(obs.mea);

    rb.param_blocks = {
        param_blocks[obs.camera_id * 2].param_ptr,
        param_blocks[obs.camera_id * 2 + 1].param_ptr,
        param_blocks[num_cam * 2 + obs.point_id].param_ptr,
    };

    rb.param_blocks_candidate = {
        param_blocks[obs.camera_id * 2].param_new.data(),
        param_blocks[obs.camera_id * 2 + 1].param_new.data(),
        param_blocks[num_cam * 2 + obs.point_id].param_new.data(),
    };
  }
}

void SolveProblem(BAGraph &ba_graph, std::vector<ParamBlock<default_float>> &param_blocks,
                  std::vector<ResidualBlock<default_float>> &residual_blocks) {
  auto &obs_vec = ba_graph.obs_vec;
  auto &cam_vec = ba_graph.cam_vec;
  auto &point_vec = ba_graph.point_vec;

  std::vector<std::pair<int, int>> edge_vec;
  edge_vec.resize(obs_vec.size());
  for (int i = 0; i < obs_vec.size(); ++i) {
    edge_vec[i] = {obs_vec[i].camera_id, obs_vec[i].point_id};
  }

  LMMinimizer lm;
  {
    lm.init(_max_iter_);
    lm.solver.edge_vec = edge_vec;
    lm.solver.param_blocks = std::move(param_blocks);
    lm.solver.residual_blocks = std::move(residual_blocks);
  }

  // Timer t4("init %.6lfs\n");
  // t4.resume();
  lm.solver.init(edge_vec.size(), cam_vec.size(), point_vec.size());
  // t4.stop_and_log();

  // Timer t5("solver %.6lfs\n");
  // t5.resume();
  lm.solve_problem();
  // t5.stop_and_log();
}

void RunCoLi(BAGraph& ba_graph) {
  std::vector<ParamBlock<default_float>> param_blocks;
  std::vector<ResidualBlock<default_float>> residual_blocks;
  AddParamblock(ba_graph, param_blocks);
  AddResidualblock(ba_graph, param_blocks, residual_blocks);
  SolveProblem(ba_graph, param_blocks, residual_blocks);
}
