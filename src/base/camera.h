
#pragma once

#include <Eigen/Eigen>

template <typename T>
inline void Distortion(const T *extra_params, const T u, const T v, T *du, T *dv) {
  const T k = extra_params[0];

  const T u2 = u * u;
  const T v2 = v * v;
  const T r2 = u2 + v2;
  const T radial = k * r2;
  *du = u * radial;
  *dv = v * radial;
}

inline void IterativeUndistortion(const double *params, double *u, double *v) {
  // Parameters for Newton iteration using numerical differentiation with
  // central differences, 100 iterations should be enough even for complex
  // camera models with higher order terms.
  const size_t kNumIterations = 100;
  const double kMaxStepNorm = 1e-10;
  const double kRelStepSize = 1e-6;

  Eigen::Matrix2d J;
  const Eigen::Vector2d x0(*u, *v);
  Eigen::Vector2d x(*u, *v);
  Eigen::Vector2d dx;
  Eigen::Vector2d dx_0b;
  Eigen::Vector2d dx_0f;
  Eigen::Vector2d dx_1b;
  Eigen::Vector2d dx_1f;

  for (size_t i = 0; i < kNumIterations; ++i) {
    const double step0 = std::max(std::numeric_limits<double>::epsilon(), std::abs(kRelStepSize * x(0)));
    const double step1 = std::max(std::numeric_limits<double>::epsilon(), std::abs(kRelStepSize * x(1)));
    Distortion(params, x(0), x(1), &dx(0), &dx(1));
    Distortion(params, x(0) - step0, x(1), &dx_0b(0), &dx_0b(1));
    Distortion(params, x(0) + step0, x(1), &dx_0f(0), &dx_0f(1));
    Distortion(params, x(0), x(1) - step1, &dx_1b(0), &dx_1b(1));
    Distortion(params, x(0), x(1) + step1, &dx_1f(0), &dx_1f(1));
    J(0, 0) = 1 + (dx_0f(0) - dx_0b(0)) / (2 * step0);
    J(0, 1) = (dx_1f(0) - dx_1b(0)) / (2 * step1);
    J(1, 0) = (dx_0f(1) - dx_0b(1)) / (2 * step0);
    J(1, 1) = 1 + (dx_1f(1) - dx_1b(1)) / (2 * step1);
    const Eigen::Vector2d step_x = J.inverse() * (x + dx - x0);
    x -= step_x;
    if (step_x.squaredNorm() < kMaxStepNorm) {
      break;
    }
  }

  *u = x(0);
  *v = x(1);
}

class CameraModel {
 public:
  double fx() const { return camera_params[0]; }
  double fy() const { return camera_params[0]; }
  double cx() const { return camera_params[1]; }
  double cy() const { return camera_params[2]; }
  void normalize_point(const Eigen::Vector2d &point, Eigen::Vector2d &point_normlized) {
    point_normlized.x() = (point.x() - cx()) / fx();
    point_normlized.y() = (point.y() - cy()) / fy();
    if (camera_params[3] != 0) {
      IterativeUndistortion(&camera_params[3], &point_normlized.x(), &point_normlized.y());
    }
  }
  int id;
  int width, height;
  std::array<double, 4> camera_params;
  void log() {
    printf("%d %d %d\n", id, width, height);
    printf("%lf %lf %lf\n", camera_params[0], camera_params[1], camera_params[2]);
  }
};
