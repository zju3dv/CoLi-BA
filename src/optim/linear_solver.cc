//
// Created by SENSETIME\yezhichao1 on 2020/4/28.
//
#include "linear_solver.h"

#include <Eigen/Eigen>
#include <iostream>

#include "utility/global.h"

namespace coli {

void linear_solver::init(int num_res, int num_cam, int num_point) {
  num_res_ = num_res, num_cam_ = num_cam, num_point_ = num_point;

  jc_blocks.assign(num_res_, JAC_BLOCK_CAM::Zero());
  jp_blocks.assign(num_res_, JAC_BLOCK_POINT::Zero());
  residuals.assign(num_res_, RES_BLOCK::Zero());
  etf_etei_s.assign(num_res_, matrix6x3::Zero());
  etes.assign(num_point_, matrix3::Zero());
  etbs.assign(num_point_, vector3::Zero());
  x_f.resize(6 * num_cam_);
  x_e.resize(3 * num_point_);

  num_block_row_ = num_cam_;
  if (const_cam_flags.empty()) {
    const_cam_flags.assign(num_cam, false);
  } else {
    for (auto flag : const_cam_flags) {
      if (flag) num_block_row_--;
    }
  }
  id_origin2real.assign(num_cam_, -1);
  int id_block = 0;
  for (int i = 0; i < num_cam_; ++i) {
    if (const_cam_flags[i]) continue;
    id_origin2real[i] = id_block;
    id_block++;
  }

  real_lhs.clear();
  real_lhs.resize(num_block_row_);
  real_rhs.resize(num_block_row_ * 6);
  real_dx.resize(num_block_row_ * 6);

  pid2rid_.assign(num_point_, std::vector<int>(0));
  cam_res_size_.resize(num_cam_ + 1, 0);
  cam_pair2res_pair_.clear();
  cam_pair2res_pair_.resize(num_cam_);

  for (int id_res = 0; id_res < edge_vec.size(); ++id_res) {
    const auto &[id_cam, id_point] = edge_vec[id_res];
    pid2rid_[id_point].emplace_back(id_res);
    if ((id_res + 1 == edge_vec.size()) || (id_cam != edge_vec[id_res + 1].first)) {
      cam_res_size_[id_cam + 1] = id_res + 1;
    }
  }

  for (int id_cam1 = 0; id_cam1 < num_cam_; ++id_cam1) {
    if (const_cam_flags[id_cam1]) continue;
    const int id_res_b = cam_res_size_[id_cam1], id_res_e = cam_res_size_[id_cam1 + 1];
    for (int id_res1 = id_res_b; id_res1 < id_res_e; ++id_res1) {
      const int id_pt = edge_vec[id_res1].second;
      for (auto &id_res2 : pid2rid_[id_pt]) {
        if (id_res2 < id_res1) continue;
        const int id_cam2 = edge_vec[id_res2].first;
        if (const_cam_flags[id_cam2]) continue;
        if (cam_pair2res_pair_[id_cam1].count(id_cam2) == 0)
          cam_pair2res_pair_[id_cam1][id_cam2] = std::vector<std::pair<int, int>>(0);
        cam_pair2res_pair_[id_cam1][id_cam2].emplace_back(std::pair<int, int>(id_res1, id_res2));
      }
    }
  }

  // for (int id_cam1 = 0; id_cam1 < num_cam_; ++id_cam1) {
  //   if (const_cam_flags[id_cam1]) continue;
  //   const int id_row = id_origin2real[id_cam1];
  //   const int id_res_b = cam_res_size_[id_cam1], id_res_e = cam_res_size_[id_cam1 + 1];
  //   for (int id_res1 = id_res_b; id_res1 < id_res_e; ++id_res1) {
  //     const int id_pt = edge_vec[id_res1].second;
  //     for (auto &id_res2 : pid2rid_[id_pt]) {
  //       if (id_res2 < id_res1) continue;
  //       const int id_cam2 = edge_vec[id_res2].first;
  //       if (const_cam_flags[id_cam2]) continue;
  //       const int id_col = id_origin2real[id_cam2];
  //       if (real_lhs[id_row].count(id_col) == 0){
  //         real_lhs[id_row][id_col] = matrix6::Zero();
  //       }
  //     }
  //   }
  // }

  using namespace ceres::internal;
  LinearSolver::Options sparse_cholesky_options;
  sparse_cholesky_options.use_postordering = true;
  sparse_cholesky_ = SparseCholesky::Create(sparse_cholesky_options);

  timer_linearization.init("Time Linearization %.7lfs\n");
  timer_compute_delta.init("Time compute delta %.7lfs\n");
  timer_schur_complement.init("Time schur complement %.7lfs\n");
  timer_vec = {&timer_linearization, &timer_schur_complement, &timer_compute_delta};
}

void linear_solver::linearization() {
  // for every residual block compute {Jacobian, residual}
  for (int i = 0; i < num_res_; ++i) {
    const auto &rb = residual_blocks[i];
    const std::array<default_float *, 2> jac_ptr = {jc_blocks[i].data(), jp_blocks[i].data()};
    rb.factor->Evaluate_Jac(rb.param_blocks.data(), residuals[i].data(), jac_ptr.data());
  }
}

void linear_solver::schur_complement() {
  // lhs += sum_i f_i^T * f_i - e_t_f^T * e_t_e_inverse * e_t_f
  // rhs += f_t_b - e_t_f^T * e_t_e_inverse * e_t_b

  // step1: compute ete, etb
  const default_float diag_val = 1.0 / radius_;
  matrix3 diag_radius = matrix3::Zero();
  diag_radius.diagonal().setConstant(diag_val);
  for (int id_pt = 0; id_pt < num_point_; ++id_pt) {
    matrix3 ete = diag_radius;
    vector3 etb = vector3::Zero();
    for (const int id_res : pid2rid_[id_pt]) {
      ete.noalias() += jp_blocks[id_res].transpose() * jp_blocks[id_res];
      etb.noalias() += jp_blocks[id_res].transpose() * residuals[id_res];
    }
    etes[id_pt] = ete.inverse();
    etbs[id_pt] = etb;
  }

  // step2: compute etf_etei = e_t_f^T * e_t_e_inverse
  for (int id_res = 0; id_res < num_res_; ++id_res) {
    const int id_pt = edge_vec[id_res].second;
    etf_etei_s[id_res] = jc_blocks[id_res].transpose() * jp_blocks[id_res] * etes[id_pt];
  }

  // for (auto&lhs_row:real_lhs){
  //   for(auto&[col_id,block_mat]:lhs_row){
  //     block_mat.setZero();
  //   }
  // }

  // step3: compute -etf_etei * e_t_f  & -etf_etei * e_t_b
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    const int id_row = id_origin2real[id_cam];
    for (const auto &[id_cam2, id_res_pairs] : cam_pair2res_pair_[id_cam]) {
      matrix6 tmp = matrix6::Zero();
      for (const auto &[id_res1, id_res2] : id_res_pairs) {
        tmp.noalias() -= etf_etei_s[id_res1] * jp_blocks[id_res2].transpose() * jc_blocks[id_res2];
      }
      const int id_col = id_origin2real[id_cam2];
      real_lhs[id_row][id_col] = tmp;
    }

    // {
    //   const int id_res_b = cam_res_size_[id_cam], id_res_e = cam_res_size_[id_cam + 1];
    //   for (int id_res1 = id_res_b; id_res1 < id_res_e; ++id_res1) {
    //     const int id_pt = edge_vec[id_res1].second;
    //     for (auto &id_res2 : pid2rid_[id_pt]) {
    //       if (id_res2 < id_res1) continue;
    //       const int id_cam2 = edge_vec[id_res2].first;
    //       if (const_cam_flags[id_cam2]) continue;
    //       const int id_col = id_origin2real[id_cam2];
    //       real_lhs[id_row][id_col]-=etf_etei_s[id_res1] * jp_blocks[id_res2].transpose() * jc_blocks[id_res2];
    //     }
    //   }
    // }

    vector6 tmpv = vector6::Zero();
    const int id_res0 = cam_res_size_[id_cam], id_res1 = cam_res_size_[id_cam + 1];
    for (int id_res = id_res0; id_res < id_res1; ++id_res) {
      const int id_pt = edge_vec[id_res].second;
      tmpv.noalias() -= etf_etei_s[id_res] * etbs[id_pt];
    }
    real_rhs.segment<6>(id_row * 6).noalias() = tmpv;
  }

  // step4:  compute ftf, ftb
  matrix6 diag_radius1 = matrix6::Zero();
  diag_radius1.diagonal().setConstant(diag_val);
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    const int id_res0 = cam_res_size_[id_cam], id_res1 = cam_res_size_[id_cam + 1];
    matrix6 ftf = diag_radius1;
    vector6 ftb = vector6::Zero();
    for (int id_res = id_res0; id_res < id_res1; ++id_res) {
      ftf.noalias() += jc_blocks[id_res].transpose() * jc_blocks[id_res];
      ftb.noalias() += jc_blocks[id_res].transpose() * residuals[id_res];
    }
    const int id_row = id_origin2real[id_cam];
    real_lhs[id_row][id_row] += ftf;
    real_rhs.segment<6>(id_row * 6).noalias() += ftb;
  }
}

void linear_solver::compute_delta() {
  // slove Ax = b
  if (linearsolver_type == DENSE_SCHUR) {
    solve_linear_system_sparse(real_lhs, real_rhs, real_dx, sparse_cholesky_.get());
  } else if (linearsolver_type == SPARSE_SCHUR) {
    solve_linear_system_sparse(real_lhs, real_rhs, real_dx, sparse_cholesky_.get());
  } else if (linearsolver_type == ITERATIVE_SCHUR) {
    solve_linear_system_pcg(real_lhs, real_rhs.data(), real_dx.data());
  }

  x_f.setZero();
  int id_real = 0;
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    x_f.segment<6>(id_cam * 6) = real_dx.segment<6>(id_real * 6);
    id_real++;
  }

  // x_e_i = e_t_e_inverse * sum_i e_i^T * (b_i - f_i * x_f);
  // x_e_i = e_t_e_inverse * sum_i (e_t_b - e_t_f * x_f);
  for (int id_pt = 0; id_pt < num_point_; ++id_pt) {
    vector3 tmp = etbs[id_pt];
    for (const int id_res : pid2rid_[id_pt]) {
      const int id_cam = edge_vec[id_res].first;
      tmp -= jp_blocks[id_res].transpose() * jc_blocks[id_res] * x_f.block<6, 1>(6 * id_cam, 0);
    }
    x_e.segment<3>(3 * id_pt) = etes[id_pt] * tmp;
  }
  x_f = -x_f;
  x_e = -x_e;
}

void linear_solver::solve_linear_problem(double radius) {
  radius_ = radius;
  TIMING(timer_linearization,linearization());
  TIMING(timer_schur_complement,schur_complement());
  TIMING(timer_compute_delta,compute_delta());
}

void linear_solver::compute_model_cost_change() {
  // new_model_cost
  //  = 1/2 [f + J * step]^2
  //  = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
  // model_cost_change
  //  = cost - new_model_cost
  //  = f'f/2  - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
  //  = -f'J * step - step' * J' * J * step / 2
  //  = -(J * step)'(f + J * step / 2)
  default_float _model_cost_change = 0;
  for (int id_res = 0; id_res < num_res_; ++id_res) {
    const auto &[id_cam, id_pt] = edge_vec[id_res];
    vector<residual_size> model_residual =
        jc_blocks[id_res] * x_f.segment<6>(6 * id_cam) + jp_blocks[id_res] * x_e.segment<3>(3 * id_pt);
    vector<residual_size> tmp = model_residual / 2 + residuals[id_res];
    _model_cost_change -= model_residual.dot(tmp);
  }
  model_cost_change_ = _model_cost_change;
  // std::cout << "model cost change: " << model_cost_change_ << std::endl;
}

void linear_solver::compute_cost(double &cost) {
  cost = 0;
  for (int i = 0; i < num_res_; ++i) {
    cost += residuals[i].squaredNorm();
  }
  cost /= 2;
}

void linear_solver::compute_relative_decrease() {
  compute_cost(current_cost_);
  // compute candidate param
  param_plus_delta();
  // compute relative_decrease
  candidate_cost_ = 0;
  for (int i = 0; i < num_res_; ++i) {
    RES_BLOCK res;
    const auto &rb = residual_blocks[i];
    rb.factor->EvaluateRes(rb.param_blocks_candidate.data(), res.data());
    candidate_cost_ += res.squaredNorm();
  }
  candidate_cost_ /= 2;
  relative_decrease_ = (current_cost_ - candidate_cost_) / model_cost_change_;
}

void linear_solver::compute_step_norm_x_norm(double &step_norm, double &x_norm) {
  step_norm = x_norm = 0;
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    step_norm += param_blocks[id_cam * 2].step_square_norm();
    step_norm += param_blocks[id_cam * 2 + 1].step_square_norm();
    x_norm += param_blocks[id_cam * 2].x_square_norm();
    x_norm += param_blocks[id_cam * 2 + 1].x_square_norm();
  }
  step_norm = sqrt(step_norm);
  x_norm = sqrt(x_norm);
}

void linear_solver::param_plus_delta() {
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    default_float *delta_x = x_f.block<6, 1>(6 * id_cam, 0).data();
    param_blocks[id_cam * 2].plus_delta(delta_x);
    param_blocks[id_cam * 2 + 1].plus_delta(delta_x + 3);
  }
  for (int id_pt = 0; id_pt < num_point_; ++id_pt) {
    default_float *delta_x = x_e.block<3, 1>(3 * id_pt, 0).data();
    param_blocks[num_cam_ * 2 + id_pt].plus_delta(delta_x);
  }
}

void linear_solver::param_update() {
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    param_blocks[id_cam * 2].update();
    param_blocks[id_cam * 2 + 1].update();
  }
  for (int id_pt = 0; id_pt < num_point_; ++id_pt) {
    param_blocks[num_cam_ * 2 + id_pt].update();
  }
}

void linear_solver::log() {
  for (auto &timer_ptr : timer_vec) timer_ptr->log();
}

}  // namespace coli
