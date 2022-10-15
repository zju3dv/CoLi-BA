
#include "base/map.h"
#include "base/map_io.hpp"
#include "base/ba_graph.h"
#include "run_ba_solver.h"
#include "utility/timer.h"

using namespace coli;
 
BAGraph LoadGraph(const std::string path, int num) {
  Map map;
  read_map(path, map);
  map.filter_multi_connection_points();
  map.reorder_id_colmap();
  map.add_noise(3e-4);
  printf("read map done\n");

  BAGraph ba_graph;
  ba_graph.ConvertFromMap(map,num);
  printf("build obs done\n");
  return ba_graph;
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  int max_num_frame;
  std::string data_path,optimizer_name; 
  if (argc == 3||argc == 4) {
      data_path = argv[1];
      optimizer_name = argv[2];
      if(argc == 4){
        max_num_frame = std::stoi(argv[3]);
      }else{
        max_num_frame = 100000;
      }
  }
  
  BAGraph ba_graph = LoadGraph(data_path, max_num_frame);
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