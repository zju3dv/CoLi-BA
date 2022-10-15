//
// Created by SENSETIME\yezhichao1 on 2020/4/28.
//
#include <Eigen/Eigen>
#include <iostream>

#include "linear_solver.h"
#include "utility/global.h"

namespace coli {

#define knum_thread 1

inline void equal_33_l(const vector3 &a, const matrix3 &H, matrix6x3 &r) {
  // r = fte*H
  // e = a'x f = A,a'x
  // fte = [a'x,-a'x*a'x]
  for (int i = 0; i < 3; ++i) {
    r.col(i).head<3>().noalias() = a.cross(H.col(i));
    r.col(i).tail<3>().noalias() = -a.cross(a.cross(H.col(i)));
  }
}

inline void sub_63_r(const vector3 &a, const matrix6x3 &H, matrix6 &r) {
  // r -= H*etf
  // etf = -a'x,-a'x*a'x
  for (int i = 0; i < 6; ++i) {
    vector3 tmp = a.cross(H.row(i));
    r.row(i).head<3>().noalias() -= tmp;
    r.row(i).tail<3>().noalias() += a.cross(tmp);
  }
}

inline void sub_61_l(const vector3 &a, const vector6 &H, vector3 &r) {
  r.noalias() += a.cross(H.head<3>() + a.cross(H.tail<3>()));
}

void linear_solver_refine::init(int num_res, int num_cam, int num_point) {
  num_res_ = num_res, num_cam_ = num_cam, num_point_ = num_point;

  factors_v3.assign(num_res_, vector3::Zero());
  factors_v4.assign(num_res_, vector4::Zero());
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

  // const_pt_flags.assign(num_point_,false);
  // const_pt_flags[0] = true;

  real_lhs.clear();
  real_lhs.resize(num_block_row_);
  real_rhs.resize(num_block_row_ * 6);
  real_dx.resize(num_block_row_ * 6);

  pid2rid_.assign(num_point_, std::vector<int>(0));
  cam_res_size_.resize(num_cam_ + 1, 0);
  cam_pair2res_pair_.clear();
  cam_pair2res_pair_.resize(num_cam_);

  int max_num_res = 0;
  for (int id_res = 0; id_res < edge_vec.size(); ++id_res) {
    const auto &[id_cam, id_point] = edge_vec[id_res];
    pid2rid_[id_point].emplace_back(id_res);
    if ((id_res + 1 == edge_vec.size()) || (id_cam != edge_vec[id_res + 1].first)) {
      cam_res_size_[id_cam + 1] = id_res + 1;
      int num_res_cam = id_res + 1 - cam_res_size_[id_cam];
      if (max_num_res < num_res_cam) max_num_res = num_res_cam;
    }
  }
  // etf_etei_s.assign(max_num_res, matrix6x3::Zero());

  int count = 0;
  int count1 = 0;
// #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_cam1 = 0; id_cam1 < num_cam_; ++id_cam1) {
    if (const_cam_flags[id_cam1]) continue;
    const int id_res_b = cam_res_size_[id_cam1], id_res_e = cam_res_size_[id_cam1 + 1];
    for (int id_res1 = id_res_b; id_res1 < id_res_e; ++id_res1) {
      const int id_pt = edge_vec[id_res1].second;
      // if(const_pt_flags[id_pt])continue;
      for (auto &id_res2 : pid2rid_[id_pt]) {
        if (id_res2 < id_res1) continue;
        // if (id_res2 <= id_res1) continue;
        const int id_cam2 = edge_vec[id_res2].first;
        if (const_cam_flags[id_cam2]) continue;
        if (cam_pair2res_pair_[id_cam1].count(id_cam2) == 0) {
          cam_pair2res_pair_[id_cam1][id_cam2] = std::vector<std::pair<int, int>>(0);
          // count++;
        }
        cam_pair2res_pair_[id_cam1][id_cam2].emplace_back(std::pair<int, int>(id_res1, id_res2));
        // count1++;
      }
    }
  }
  // std::cout<<count<<" "<<count1<<std::endl;

  using namespace ceres::internal;
  LinearSolver::Options sparse_cholesky_options;
  sparse_cholesky_options.use_postordering = true;
  sparse_cholesky_ = SparseCholesky::Create(sparse_cholesky_options);

  timer_linearization.init("Time Linearization %.7lfs\n");
  timer_compute_delta.init("Time compute delta %.7lfs\n");
  timer_schur_complement.init("Time schur complement %.7lfs\n");
  timer_vec = {&timer_linearization, &timer_schur_complement, &timer_compute_delta};
}

void linear_solver_refine::linearization() {
  // for every residual block compute {Jacobian, residual}
// #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int i = 0; i < num_res_; ++i) {
    const auto &rb = residual_blocks[i];
    const std::array<default_float *, 2> factors_ptr = {factors_v3[i].data(), factors_v4[i].data()};
    rb.factor->Evaluate_Fac(rb.param_blocks.data(), residuals[i].data(), factors_ptr.data());
  }
}

void linear_solver_refine::schur_complement() {
  // lhs += sum_i f_i^T * f_i - e_t_f^T * e_t_e_inverse * e_t_f
  // rhs += f_t_b - e_t_f^T * e_t_e_inverse * e_t_b

  // step1: compute ete, etb
  const default_float diag_val = 1.0 / radius_;
  matrix3 diag_radius = matrix3::Zero();
  diag_radius.diagonal().setConstant(diag_val);
// #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_pt = 0; id_pt < num_point_; ++id_pt) {
    matrix3 ete = diag_radius;
    vector3 etb = vector3::Zero();
    for (const int id_res : pid2rid_[id_pt]) {
      const_map<vector3> a(factors_v4[id_res].data());
      const default_float s = factors_v4[id_res](3);
      const default_float data[] = {1.0 - a(0) * a(0), -a(0) * a(1), -a(0) * a(2), -a(1) * a(0),     1.0 - a(1) * a(1),
                                    -a(1) * a(2),      -a(2) * a(0), -a(2) * a(1), 1.0 - a(2) * a(2)};
      const_map<matrix3> A(data);
      ete.noalias() += (s * s) * A;
      etb.noalias() += s * (residuals[id_res] - (a.dot(residuals[id_res])) * a);
    }
    etes[id_pt] = ete.inverse();
    etbs[id_pt] = etb;
  }


// step2: compute etf_etei = e_t_f^T * e_t_e_inverse
// #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_res = 0; id_res < num_res_; ++id_res) {
    const int id_pt = edge_vec[id_res].second;
    equal_33_l(factors_v3[id_res], etes[id_pt], etf_etei_s[id_res]);
  }

// step3: compute -etf_etei * e_t_f  & -etf_etei * e_t_b
// #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    const int id_row = id_origin2real[id_cam];
    for (const auto &[id_cam2, id_res_pairs] : cam_pair2res_pair_[id_cam]) {
      matrix6 tmp = matrix6::Zero();
      for (const auto &[id_res1, id_res2] : id_res_pairs) {
        sub_63_r(factors_v3[id_res2], etf_etei_s[id_res1], tmp);
      }
      const int id_col = id_origin2real[id_cam2];
      real_lhs[id_row][id_col] = tmp;
    }
    vector6 tmpv = vector6::Zero();
    const int id_res0 = cam_res_size_[id_cam], id_res1 = cam_res_size_[id_cam + 1];
    for (int id_res = id_res0; id_res < id_res1; ++id_res) {
      const int id_pt = edge_vec[id_res].second;
      // if(const_pt_flags[id_pt])continue;
      tmpv.noalias() -= etf_etei_s[id_res] * etbs[id_pt];
    }
    real_rhs.segment<6>(id_row * 6).noalias() = tmpv;
  }

// another implementation for step2+step3 for saving memory 
// for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
//   if (const_cam_flags[id_cam]) continue;
//   const int id_row = id_origin2real[id_cam];

//   const int id_res0 = cam_res_size_[id_cam], id_res1 = cam_res_size_[id_cam + 1];
//   vector6 tmpv = vector6::Zero();
//   for (int id_res = id_res0; id_res < id_res1; ++id_res) {
//     const int id_pt = edge_vec[id_res].second;
//     equal_33_l(factors_v3[id_res], etes[id_pt], etf_etei_s[id_res-id_res0]);
//     tmpv.noalias() -= etf_etei_s[id_res-id_res0] * etbs[id_pt];
//   }
//   real_rhs.segment<6>(id_row * 6).noalias() = tmpv;

//   for (const auto &[id_cam2, id_res_pairs] : cam_pair2res_pair_[id_cam]) {
//     matrix6 tmp = matrix6::Zero();
//     for (const auto &[id_res1, id_res2] : id_res_pairs) {
//       sub_63_r(factors_v3[id_res2], etf_etei_s[id_res1-id_res0], tmp);
//     }
//     const int id_col = id_origin2real[id_cam2];
//     real_lhs[id_row][id_col] = tmp;
//   }
// }
 
// step4:  compute ftf, ftb
// #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    const int id_res0 = cam_res_size_[id_cam], id_res1 = cam_res_size_[id_cam + 1];
    matrix3 ftf1 = diag_radius, ftf2 = diag_radius;
    vector3 ftf3 = vector3::Zero();
    vector6 ftb = vector6::Zero();
    for (int id_res = id_res0; id_res < id_res1; ++id_res) {
      const_map<vector3> a(factors_v4[id_res].data());
      const default_float &s = factors_v4[id_res](3);
      const default_float data[] = {1 - a(0) * a(0), -a(0) * a(1), -a(0) * a(2), -a(1) * a(0),   1 - a(1) * a(1),
                                    -a(1) * a(2),    -a(2) * a(0), -a(2) * a(1), 1 - a(2) * a(2)};
      const_map<matrix3> A(data);

      ftf1.noalias() += A;
      ftf2.noalias() += (s * s) * A;
      ftf3.noalias() += s * a;

      ftb.head<3>().noalias() += a.cross(residuals[id_res]);
      ftb.tail<3>().noalias() += s * (residuals[id_res] - a.dot(residuals[id_res]) * a);
    }

    matrix6 ftf;
    ftf.topLeftCorner<3, 3>().noalias() = ftf1;
    ftf.topRightCorner<3, 3>().noalias() = skewSymmetric(ftf3);
    ftf.bottomLeftCorner<3, 3>().noalias() = -skewSymmetric(ftf3);
    ftf.bottomRightCorner<3, 3>().noalias() = ftf2;
    const int id_row = id_origin2real[id_cam];
    real_lhs[id_row][id_row] += ftf;
    real_rhs.segment<6>(id_row * 6).noalias() += ftb;
  }
}

void linear_solver_refine::compute_delta() {
  // slove Ax = b
  // solve_linear_system_dense( real_lhs, real_rhs, real_dx);
  // solve_linear_system_sparse(real_lhs, real_rhs, real_dx, sparse_cholesky_.get());
  solve_linear_system_pcg(real_lhs, real_rhs.data(), real_dx.data());
  x_f.setZero();
  int id_real = 0;
  // #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_cam = 0; id_cam < num_cam_; ++id_cam) {
    if (const_cam_flags[id_cam]) continue;
    x_f.segment<6>(id_cam * 6) = real_dx.segment<6>(id_real * 6);
    id_real++;
  }

  // x_e_i = e_t_e_inverse * sum_i e_i^T * (b_i - f_i * x_f);
  // x_e_i = e_t_e_inverse * sum_i (e_t_b - e_t_f * x_f);
  // #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_pt = 0; id_pt < num_point_; ++id_pt) {
    // if(const_pt_flags[id_pt]){
    //   x_e.segment<3>(3 * id_pt).setZero();
    // }else{
    vector3 tmp = etbs[id_pt];
    for (const int id_res : pid2rid_[id_pt]) {
      const int id_cam = edge_vec[id_res].first;
      sub_61_l(factors_v3[id_res], x_f.segment<6>(6 * id_cam), tmp);
    }
    x_e.segment<3>(3 * id_pt) = etes[id_pt] * tmp;
    // }
  }
  x_f = -x_f;
  x_e = -x_e;
}

void linear_solver_refine::solve_linear_problem(double radius) {
  radius_ = radius;
  timer_linearization.resume();
  linearization();
  timer_linearization.stop();

  timer_schur_complement.resume();
  schur_complement();
  timer_schur_complement.stop();

  timer_compute_delta.resume();
  compute_delta();
  timer_compute_delta.stop();
}

void linear_solver_refine::compute_model_cost_change() {
  // new_model_cost
  //  = 1/2 [f + J * step]^2
  //  = 1/2 [ f'f + 2f'J * step + step' * J' * J * step ]
  // model_cost_change
  //  = cost - new_model_cost
  //  = f'f/2  - 1/2 [ f'f + 2f'J * step + step' * J' * J * step]
  //  = -f'J * step - step' * J' * J * step / 2
  //  = -(J * step)'(f + J * step / 2)
  default_float _model_cost_change = 0;
  // #pragma omp parallel for schedule(dynamic, knum_thread)
  for (int id_res = 0; id_res < num_res_; ++id_res) {
    const auto &[id_cam, id_pt] = edge_vec[id_res];
    const_map<vector3> a(factors_v4[id_res].data());
    const default_float s = factors_v4[id_res].data()[3];
    const vector3 b = x_f.segment<3>(6 * id_cam + 3) + x_e.segment<3>(3 * id_pt);
    vector<residual_size> model_residual = (s * (b - a.dot(b) * a)) - a.cross(x_f.segment<3>(6 * id_cam));
    vector<residual_size> tmp = model_residual / 2 + residuals[id_res];
    _model_cost_change -= model_residual.dot(tmp);
  }
  model_cost_change_ = _model_cost_change;
  // std::cout << "model cost change: " << model_cost_change_ << std::endl;
}

}  // namespace coli
