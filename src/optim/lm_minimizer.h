
#pragma once
#define USE_COLI
#include "linear_solver.h"

namespace coli {

class LMMinimizer {
 public:
  LMMinimizer() = default;

  void init(int _max_iter = 20);
  void solve_problem();

#ifdef USE_COLI
  linear_solver_refine solver;
#else
  linear_solver solver;
#endif

 private:
  void StepRejected(double step_quality);

  void StepAccepted(double step_quality);

  bool ParameterToleranceReached(double step_norm, double x_norm);

  bool FunctionToleranceReached(double x_cost, double candidate_cost);

  int max_iter = 100;

  double radius;
  double decrease_factor;
  double parameter_tolerance;
  double function_tolerance;
};
}  // namespace coli
