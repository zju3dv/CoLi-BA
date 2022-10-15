
#pragma once

#include "map.h"
#include "utility/global.h"

#define USE_REORDERING

namespace coli {

struct OBS {
  OBS() {
    camera_id = -1;
    point_id = -1;
    mea.setZero();
  }
  OBS(int cid, int pid, vector2 _mea, double _w = 1.0) {
    camera_id = cid;
    point_id = pid;
    mea = _mea;
    w = _w;
  }

  int camera_id;
  int point_id;
  vector2 mea;
  double w;
};

class BAGraph {
 public:
  enum SolverType { Dense, Sparse, Iterative };
  const bool use_ntwc = true;
  const bool auto_diff = false;
  SolverType solve_type = SolverType::Dense;

  std::vector<coli::OBS> obs_vec;
  std::vector<coli::vector<7>> cam_vec;
  std::vector<coli::vector<3>> point_vec;  
 
  inline default_float LogMSE() {
    int count = 0;
    default_float mse = 0;
    for (const auto &obs : obs_vec) {
      const auto &cam = cam_vec[obs.camera_id];
      const auto &pw = point_vec[obs.point_id];
      const double w = obs.w;
      const coli::const_map<coli::quaternion> qcw(cam.data());
      const coli::const_map<coli::vector3> t(cam.data() + 4);
      const coli::vector3 pc = use_ntwc ? qcw * (pw + t) : qcw * pw + t;
      const coli::vector2 residual = w * (pc.hnormalized() - obs.mea);
      mse += residual.norm();
      count++;
    }
    // printf("num_camera: %zu, num_point: %zu\n", cam_vec.size(), point_vec.size());
    // printf("num_obs: %d, avg_mse: %lf\n", count, mse / count);
    printf("avg_mse: %lf\n", mse / count);
    return mse / count;
  }

  inline std::map<int, int> solverTSP(std::vector<std::map<int, int>> &edge) {
    const int num_node = edge.size();
    std::vector<int> new_order(num_node, -1);
    new_order[0] = 0;
    int last_node = 0;
    for (int iter = 1; iter < num_node; ++iter) {
      // find nearst neighbor of last node
      auto &edges = edge[last_node];
      int near_dist = -1;
      int near_id = -1;
      for (int i = 0; i < num_node; ++i) {
        if (new_order[i] != -1) continue;
        int dist = 0;
        if (i < last_node) {
          dist = edge[i].count(last_node) == 0 ? 0 : edge[i].at(last_node);
        } else {
          dist = edges.count(i) == 0 ? 0 : edges.at(i);
        }

        if (dist != 0) {
          if (near_dist < dist) {
            near_dist = dist;
            near_id = i;
          }
        } else if (near_id == -1) {
          near_id = i;
        }
      }
      assert(new_order[near_id] == -1);
      new_order[near_id] = iter;
      last_node = near_id;
    }
    std::map<int, int> reorder;
    for (int i = 0; i < new_order.size(); ++i) {
      reorder[i] = new_order[i];
    }
    return reorder;
  }

  inline coli::vector<7> Pose2vector7(const Pose Tcw){
      coli::vector<7> cam;
      const coli::quaternion qcw_ = Tcw.quaterniond();
      const coli::vector3 tcw_ = Tcw.t.cast<default_float>();
      const coli::vector3 ntwc_ = (qcw_.inverse() * tcw_).cast<default_float>();
      cam << qcw_.coeffs(), use_ntwc ? ntwc_ : tcw_;
      return cam;
  }

  inline void subMap(Map &map, const int num_frame_optim) {
    std::map<int, Frame> frames;
    std::map<int, Track> tracks;
    int count_frames = 0;
    int count_tracks = 0;
    for (auto &[id, frame] : map.m_frames) {
      assert(frame.registered);
      assert(count_frames == frame.id);
      if (count_frames >= num_frame_optim) break;
      frames[id] = frame;
      count_frames++;
    }
    for (auto &[id, track] : map.m_tracks) {
      int obs_num = 0;
      for (int k = 0; k < num_frame_optim; ++k) {
        if (track.m_observations.count(k) == 0) continue;
        obs_num++;
      }
      if (obs_num < 2) continue;
      std::map<int, size_t> obs;
      for (auto &[id1, id2] : track.m_observations) {
        if (id1 < num_frame_optim) {
          obs[id1] = id2;
        }
      }
      track.m_observations = obs;
      tracks[id] = track;
      count_tracks++;
    }
    for (auto &[id, frame] : frames) {
      for (int i = 0; i < frame.m_track_ids.size(); ++i) {
        int track_id = frame.m_track_ids[i];
        if (tracks.count(track_id) == 0) {
          frame.m_track_ids[i] = -1;
        }
      }
    }
    map.m_frames = frames;
    map.m_tracks = tracks;
  }

  inline void ConvertFromMap(Map &map, const int _num_frame_optim) {
    const int num_frame_optim = std::min(_num_frame_optim, int(map.m_frames.size()));
    if (num_frame_optim < map.m_frames.size()) subMap(map, num_frame_optim);
    CHECK(num_frame_optim == map.m_frames.size());
    
    // init
    std::map<int, int> id_cam_reorder2ori;
    std::map<int, int> id_pt_reorder2ori;
    std::map<int, int> id_pt_ori2reorder;
    id_cam_reorder2ori.clear();
    id_pt_reorder2ori.clear();
    id_pt_ori2reorder.clear();
    cam_vec.reserve(num_frame_optim);
    int count_point = 0, count_camera = 0;

#ifdef USE_REORDERING
    auto covisiblity_graph = map.covisiblity_graph(num_frame_optim);
    std::map<int, int> reorder_cam = solverTSP(covisiblity_graph);
    map.ReorderFrames(reorder_cam);
#else
    for (auto &[id, track] : map.m_tracks) {
      int obs_num = 0;
      for (int k = 0; k < num_frame_optim; ++k) {
        if (track.m_observations.count(k) == 0) continue;
        obs_num++;
      }
      if (obs_num < 2) continue;
      id_pt_reorder2ori[count_point] = id;
      id_pt_ori2reorder[id] = count_point;
      count_point++;
      const coli::vector3 p3d = track.m_point3d.cast<default_float>();
      point_vec.emplace_back(p3d);
    }
#endif
    
    for (auto &[id, frame] : map.m_frames) {
      assert(frame.registered);
      assert(count_camera == frame.id);
      if (count_camera >= num_frame_optim) break;
      // focal
      const double focal = frame.m_camera_model->fx();
      // add camera 
      cam_vec.push_back(Pose2vector7(frame.Tcw));
      // id_cam_reorder2ori[count_camera] = frame.id;
      const int camera_id_reorder = count_camera++; 
      // 3d point & observation
      for (int i = 0; i < frame.m_points.size(); ++i) {
        const int track_id = frame.m_track_ids[i];
        if (track_id == -1) continue;
        Track &track = map.m_tracks[track_id]; 
        if (track.num_visible_frame(num_frame_optim) < 2) continue;
        // add point3d
        if (id_pt_ori2reorder.count(track_id) == 0) {
          id_pt_reorder2ori[count_point] = track_id;
          id_pt_ori2reorder[track_id] = count_point;
          point_vec.emplace_back(track.m_point3d.cast<default_float>());
          count_point++; 
        }
        const int point_id_reorder = id_pt_ori2reorder[track_id];
        // add observation
        const coli::vector2 mea = frame.m_points_normalized[i].cast<default_float>();
        obs_vec.emplace_back(camera_id_reorder, point_id_reorder, mea, focal);
      }
    }

    sort(obs_vec.begin(), obs_vec.end(), [&id_pt_reorder2ori](const coli::OBS &l, const coli::OBS &r) {
      if (l.camera_id < r.camera_id) {
        return true;
      } else if (l.camera_id == r.camera_id) {
        if (l.point_id < r.point_id) return true;
      }
      return false;
    });
  }
};

}  // namespace coli
