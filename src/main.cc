
#include "base/ba_graph.h"
#include "base/map.h"
#include "base/map_io.hpp"
#include "run_ba_solver.h"
#include "utility/timer.h"

using namespace coli;

BAGraph LoadGraph(const std::string path, double noise, int num) {
  Map map;
  read_map(path, map);
  map.filter_multi_connection_points();
  map.reorder_id_colmap();
  map.add_noise(noise);
  // printf("read map done\n");

  BAGraph ba_graph;
  ba_graph.ConvertFromMap(map, num);
  printf("build obs done\n");
  return ba_graph;
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  int max_num_frame = 100000;
  double noise_intensity = 1e-4;
  std::string data_path, optimizer_name;
  if (argc >= 3 && argc <= 5) {
    optimizer_name = argv[1];
    data_path = argv[2];
    if (argc >= 4) {
      noise_intensity = std::stod(argv[3]);
      CHECK(noise_intensity <= 1 && noise_intensity >= 0);
    }
    if (argc == 5) {
      max_num_frame = std::stoi(argv[4]);
      CHECK(max_num_frame >= 0);
    }
  } else {
    std::cout << "Command not recognized!" << std::endl;
  }

  BAGraph ba_graph = LoadGraph(data_path, noise_intensity, max_num_frame);
  ba_graph.solve_type = BAGraph::SolverType::Iterative;

  ba_graph.LogMSE();
  Timer total_timer("Time Total %.6lfs\n");
  total_timer.start();
  if (optimizer_name == "ceres") {
    RunCeres(ba_graph);
  } else if (optimizer_name == "g2o") {
    RunG2O(ba_graph);
  } else if (optimizer_name == "coli") {
    RunCoLi(ba_graph);
  }
  total_timer.stop_and_log();
  ba_graph.LogMSE();
}