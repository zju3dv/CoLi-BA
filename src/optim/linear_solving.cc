#include <Eigen/Eigen>
#include <iostream>

#include "ceres/block_random_access_diagonal_matrix.h"
#include "ceres/block_random_access_sparse_matrix.h"
#include "ceres/conjugate_gradients_solver.h"
#include "linear_solver.h"
#include "utility/global.h"

namespace ceres::internal {

using namespace coli;

class BlockRandomAccessSparseMatrixAdapter : public LinearOperator {
 public:
  explicit BlockRandomAccessSparseMatrixAdapter(const BlockRandomAccessSparseMatrix &m) : m_(m) {}

  virtual ~BlockRandomAccessSparseMatrixAdapter() {}

  // y = y + Ax;
  void RightMultiply(const double *x, double *y) const final { m_.SymmetricRightMultiply(x, y); }

  // y = y + A'x;
  void LeftMultiply(const double *x, double *y) const final { m_.SymmetricRightMultiply(x, y); }

  int num_rows() const final { return m_.num_rows(); }
  int num_cols() const final { return m_.num_rows(); }

 private:
  const BlockRandomAccessSparseMatrix &m_;
};

class BlockRandomAccessDiagonalMatrixAdapter : public LinearOperator {
 public:
  explicit BlockRandomAccessDiagonalMatrixAdapter(const BlockRandomAccessDiagonalMatrix &m) : m_(m) {}

  virtual ~BlockRandomAccessDiagonalMatrixAdapter() {}

  // y = y + Ax;
  void RightMultiply(const double *x, double *y) const final { m_.RightMultiply(x, y); }

  // y = y + A'x;
  void LeftMultiply(const double *x, double *y) const final { m_.RightMultiply(x, y); }

  int num_rows() const final { return m_.num_rows(); }
  int num_cols() const final { return m_.num_rows(); }

 private:
  const BlockRandomAccessDiagonalMatrix &m_;
};

inline void ConvertToBRSM(const SparseBlockStorage &lhs, BlockRandomAccessSparseMatrix *sc) {
  const int num_blocks = lhs.size();
  for (int id_row_block = 0; id_row_block < num_blocks; ++id_row_block) {
    for (auto &[id_col_block, block_matrix] : lhs.at(id_row_block)) {
      int row, col, rs, cs;
      CellInfo *cell_info = sc->GetCell(id_row_block, id_col_block, &row, &col, &rs, &cs);
      coli::map<coli::matrix6> cell(cell_info->values);
      cell = block_matrix.transpose();
    }
  }
}

inline CompressedRowSparseMatrix *ConvertToCRSM(const int num_block, const SparseBlockStorage &A) {
  const int num_rows = 6 * num_block, num_cols = 6 * num_block;
  int num_nozero_block_ = 0;
  for (int id_row_block = 0; id_row_block < num_block; ++id_row_block) {
    num_nozero_block_ += A[id_row_block].size();
  }
  CompressedRowSparseMatrix *output = new CompressedRowSparseMatrix(num_rows, num_cols, num_nozero_block_ * 36);
  int *output_rows = output->mutable_rows();
  int *output_cols = output->mutable_cols();
  double *output_values = output->mutable_values();

  int id = 0;
  output_rows[0] = 0;
  for (int id_row_block = 0; id_row_block < num_block; ++id_row_block) {
    for (int r = 0; r < 6; ++r) {
      const int row_id = 6 * id_row_block + r;
      for (auto &[id_col_block, block_matrix] : A[id_row_block]) {
        for (int c = 0; c < 6; ++c) {
          const int col_id = 6 * id_col_block + c;
          output_cols[id] = col_id;
          output_values[id] = block_matrix(r, c);
          id++;
        }
      }
      output_rows[row_id + 1] = output_rows[row_id] + A[id_row_block].size() * 6;
    }
  }

  return output;
}

}  // namespace ceres::internal

namespace coli {

void linear_solver::solve_linear_system_dense(const SparseBlockStorage &lhs, vectorX &rhs, vectorX &dx) {
  const int num_block = lhs.size();
  matrixX dense_lhs = matrixX::Zero(num_block * 6, num_block * 6);
  for (int id_row_block = 0; id_row_block < num_block; ++id_row_block) {
    for (auto &[id_col_block, block_matrix] : lhs[id_row_block]) {
      dense_lhs.block<6, 6>(id_row_block * 6, id_col_block * 6) = block_matrix;
    }
  }
  Eigen::LLT<matrixX, Eigen::Upper> llt = dense_lhs.selfadjointView<Eigen::Upper>().llt();
  dx = llt.solve(rhs);
}

void linear_solver::solve_linear_system_sparse(const SparseBlockStorage &lhs, vectorX &rhs, vectorX &dx,
                                               ceres::internal::SparseCholesky *sparse_cholesky_) {
  using namespace ceres::internal;
  const int num_block = lhs.size();
  std::unique_ptr<CompressedRowSparseMatrix> _lhs;
  _lhs.reset(ConvertToCRSM(num_block, lhs));
  _lhs->set_storage_type(CompressedRowSparseMatrix::UPPER_TRIANGULAR);
  std::string message;
  sparse_cholesky_->FactorAndSolve(_lhs.get(), rhs.data(), dx.data(), &message);  // 1000 4s
}

void linear_solver::solve_linear_system_pcg(SparseBlockStorage &lhs, double *rhs, double *dx) {
  using namespace ceres::internal;
  using MatrixRef = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using VectorRef = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>;

  std::unique_ptr<ceres::internal::BlockRandomAccessSparseMatrix> sc;
  if (!sc) {
    const int num_blocks = lhs.size();
    std::vector<int> blocks_(num_blocks, 6);
    std::set<std::pair<int, int>> block_pairs;
    for (int id_row_block = 0; id_row_block < num_blocks; ++id_row_block) {
      for (auto &[id_col_block, block_matrix] : lhs[id_row_block]) {
        block_pairs.insert(std::make_pair(id_row_block, id_col_block));
      }
    }
    sc.reset(new BlockRandomAccessSparseMatrix(blocks_, block_pairs));
  }
  ConvertToBRSM(lhs, sc.get());

  const int num_blocks = lhs.size();
  const int num_rows = 6 * num_blocks;
  std::vector<int> blocks_(num_blocks, 6);
  // Size of the blocks in the Schur complement.
  std::unique_ptr<BlockRandomAccessDiagonalMatrix> preconditioner_;
  preconditioner_.reset(new BlockRandomAccessDiagonalMatrix(blocks_));

  // Extract block diagonal from the Schur complement to construct the
  // schur_jacobi preconditioner.
  for (int i = 0; i < blocks_.size(); ++i) {
    const int block_size = blocks_[i];
    int sc_r, sc_c, sc_row_stride, sc_col_stride;
    CellInfo *sc_cell_info = sc->GetCell(i, i, &sc_r, &sc_c, &sc_row_stride, &sc_col_stride);
    CHECK(sc_cell_info != nullptr);
    MatrixRef sc_m(sc_cell_info->values, sc_row_stride, sc_col_stride);
    int pre_r, pre_c, pre_row_stride, pre_col_stride;
    CellInfo *pre_cell_info = preconditioner_->GetCell(i, i, &pre_r, &pre_c, &pre_row_stride, &pre_col_stride);
    CHECK(pre_cell_info != nullptr);
    MatrixRef pre_m(pre_cell_info->values, pre_row_stride, pre_col_stride);
    pre_m.block(pre_r, pre_c, block_size, block_size) = sc_m.block(sc_r, sc_c, block_size, block_size);
  }
  preconditioner_->Invert();

  VectorRef(dx, num_rows).setZero();

  std::unique_ptr<LinearOperator> lhs_adapter(new BlockRandomAccessSparseMatrixAdapter(*sc));
  std::unique_ptr<LinearOperator> preconditioner_adapter(new BlockRandomAccessDiagonalMatrixAdapter(*preconditioner_));

  LinearSolver::Options cg_options;
  cg_options.min_num_iterations = 0;
  cg_options.max_num_iterations = 100;
  ConjugateGradientsSolver cg_solver(cg_options);

  LinearSolver::PerSolveOptions cg_per_solve_options;
  cg_per_solve_options.r_tolerance = -1;
  cg_per_solve_options.q_tolerance = 0.1;
  cg_per_solve_options.preconditioner = preconditioner_adapter.get();

  cg_solver.Solve(lhs_adapter.get(), rhs, cg_per_solve_options, dx);
}
}  // namespace coli
