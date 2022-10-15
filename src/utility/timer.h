//
// Created by SENSETIME\yezhichao1 on 2020/4/22.
//
#pragma once

#include <chrono>
#include <string>
#include <utility>
namespace coli {
#define TIMING(a, b)                                                           \
  {                                                                          \
      a.resume();                                                            \
      b;                                                                     \
      a.stop();                                                              \
  };
class Timer {
 private:
  using _Clock = std::chrono::high_resolution_clock;
  std::string format;
  double time_count;
  std::chrono::time_point<_Clock> start_time;

 public:
  Timer() : time_count(0){};
  Timer(std::string _format) : format(std::move(_format)), time_count(0){};

  void init(std::string _format) { format = std::move(_format); }

  void start() {
    time_count = 0;
    start_time = _Clock::now();
  };

  void stop() {
    auto stop_time = _Clock::now();
    time_count += std::chrono::duration_cast<std::chrono::duration<double>>(stop_time - start_time).count();
  }

  void resume() { start_time = _Clock::now(); }

  void log() { printf(format.c_str(), time_count); };

  void stop_and_log() {
    stop();
    log();
  }
};
}  // namespace coli
