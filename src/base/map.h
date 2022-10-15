//
// Created by yzc on 18-9-7.
//
#pragma once
#include <glog/logging.h>

#include <Eigen/Eigen>
#include <algorithm>
#include <iostream>

#include "camera.h"
#include "utility/global.h"

struct Pose{
  inline Eigen::Quaterniond quaterniond() const { 
    Eigen::Quaterniond _q(q(0), q(1), q(2), q(3));
    return _q;
  }
  Eigen::Vector4d q;
  Eigen::Vector3d t;
};

class Track {
 public:
  int id;
  Eigen::Vector3d m_point3d;
  std::map<int, size_t> m_observations;
  char color[3];
  double error;
  inline int num_visible_frame(const int num_frame_max){
     int obs_num = 0;
    for (int k = 0; k < num_frame_max; ++k) {
      if ( m_observations.count(k) == 0) continue; 
      obs_num++;
    } 
    return obs_num;
  }
};

class Frame {
 public:
  Frame() {}

  inline Eigen::Matrix4d twc() const {
    Eigen::Quaterniond _quat(Tcw.q(0), Tcw.q(1), Tcw.q(2), Tcw.q(3));
    Eigen::Matrix3d R = _quat.toRotationMatrix();
    Eigen::Matrix4d _Tcw;
    _Tcw << R(0, 0), R(0, 1), R(0, 2), Tcw.t(0), R(1, 0), R(1, 1), R(1, 2), Tcw.t(1), R(2, 0), R(2, 1), R(2, 2), Tcw.t(2), 0,
        0, 0, 1;
    Eigen::Matrix4d Twc = _Tcw.inverse();
    return Twc;
  }

  inline Eigen::Quaterniond quaterniond() const {
    Eigen::Quaterniond q(Tcw.q(0), Tcw.q(1), Tcw.q(2), Tcw.q(3));
    return q;
  }

  inline void set_rotation(const Eigen::Quaterniond &q) { Tcw.q << q.w(), q.x(), q.y(), q.z(); }

  inline void set_center(const Eigen::Vector3d &center) { Tcw.t = -(quaterniond() * center).transpose(); }

  inline Eigen::Vector3d center() const {
    auto Twc = twc();
    Eigen::Vector3d center;
    center << Twc(0, 3), Twc(1, 3), Twc(2, 3);
    return center;
  }

  void nomalize_point(const Eigen::Vector2d &point, Eigen::Vector2d &point_normlized) {
    m_camera_model->normalize_point(point, point_normlized);
  }

  int id;
  std::string name;
  CameraModel *m_camera_model;

  std::vector<int> m_track_ids;  // measurement to track assert -1 for no match
  std::vector<Eigen::Vector2d> m_points;
  std::vector<Eigen::Vector2d> m_points_normalized;

  Pose Tcw; 

  bool registered;
};

class Match {
 public:
  Match(int _id1 = 0, int _id2 = 0, double _dist = 0) : id1(_id1), id2(_id2), distance(_dist) {}

  int id1;
  int id2;
  double distance;
};

class Map {
 public:
  std::map<int, CameraModel> m_camera_models;
  std::map<int, Frame> m_frames;
  std::map<int, Track> m_tracks; 

  Map() {
    m_tracks.clear();
    m_frames.clear();
  }
  
  inline void RPE() {
    int count = 0;
    double mse = 0;
    for (const auto &[id, frame] : m_frames) {
      const auto q = frame.quaterniond();
      const auto t = frame.Tcw.t;
      CHECK(!frame.m_points_normalized.empty());
      for (int i = 0; i < frame.m_track_ids.size(); ++i) {
        int track_id = frame.m_track_ids.at(i);
        if (track_id == -1) continue;
        auto &p2d = frame.m_points_normalized[i];
        CHECK(m_tracks.count(track_id) != 0);
        auto &p3d = m_tracks[track_id].m_point3d;
        auto p = q * p3d + t;

        double error = (p.hnormalized() - p2d).squaredNorm();
        mse += error;
        count++;
      }
    }
    printf("RPE: %f\n", std::sqrt(mse / count));
  }

  inline void ReorderFrames(const std::map<int, int> &id_old2new) {
    std::map<int, Frame> frames;
    for (auto &[id, frame] : m_frames) {
      // std::cout<<id<<" "<<id_old2new.size() <<std::endl;
      assert(id_old2new.count(id) != 0);
      const int new_id = id_old2new.at(id);
      frame.id = new_id;
      frames[new_id] = frame;
    }
    m_frames = frames;
    for (auto &[id, track] : m_tracks) {
      std::map<int, size_t> obs;
      for (auto &[frame_id, p2d_id] : track.m_observations) {
        obs[id_old2new.at(frame_id)] = p2d_id;
      }
      track.m_observations = obs;
    }
  }

  inline void filter_multi_connection_points() {
    // ensure one 3D point only connect one 2D point in a frame
    // filter multiple 2d points in a image connect the same 3d point
    for (auto &[id, frame] : m_frames) {
      for (int i = 0; i < frame.m_track_ids.size(); ++i) {
        const int track_id = frame.m_track_ids[i];
        if (track_id == -1) continue;
        if (m_tracks.count(track_id) == 0) {
          std::cout << id << " " << i << " " << track_id << std::endl;
        } else {
          auto &track = m_tracks[track_id];
          if (track.m_observations.count(id) == 0) {
            std::cout << id << " " << track_id << std::endl;
            for (auto &[fid, pid] : track.m_observations) {
              std::cout << fid << " " << pid << std::endl;
            }
          }
          assert(track.m_observations.count(id) != 0);
          if (track.m_observations.at(id) != i) {
            frame.m_track_ids[i] = -1;
          }
        }
      }
    }
  }

  inline void reorder_id_colmap() {
    // reorder frame id
    // colmap use map storing elements
    // we use vector  storing elements)
    std::map<int, int> id_colmap2order;
    int order_id = 0;
    std::map<int, Frame> frames;
    for (auto &[id, frame] : m_frames) {
      id_colmap2order[id] = order_id;
      order_id++;
    }
    ReorderFrames(id_colmap2order);
  }

  inline void normalized_rotation() {
    for (auto &[id, frame] : m_frames) {
      if (!frame.registered) continue;
      CHECK(frame.Tcw.q.norm() != 0);
      frame.Tcw.q /= frame.Tcw.q.norm();
    }
  }

  inline void set_normalized_points() {
    for (auto &[id, frame] : m_frames) {
      if (!frame.registered) continue;
      frame.m_points_normalized.resize(frame.m_points.size());
      for (int i = 0; i < frame.m_points.size(); ++i) {
        frame.nomalize_point(frame.m_points[i], frame.m_points_normalized[i]);
      }
    }
  }

  inline void add_noise(double noise_intensity) {
    srand(0);
    for (auto &[id, frame] : m_frames) {
      frame.Tcw.q += noise_intensity * coli::vector4::Random();
      frame.Tcw.q.normalize();
      frame.Tcw.t += noise_intensity * coli::vector3::Random();
    }
    for (auto &[id, track] : m_tracks) {
      track.m_point3d += noise_intensity * coli::vector3::Random();
    }
  }

  inline std::vector<std::map<int, int>> covisiblity_graph(const int num_frame_optim){
    std::vector<std::map<int, int>> covisible(num_frame_optim); // id0 id1 weight
    std::map<int, std::vector<int>> id_pt2cams;
    for (auto &[id, track] :  m_tracks) {
      int obs_num = 0;
      std::vector<int> cid_vec(0);
      for (int k = 0; k < num_frame_optim; ++k) {
        if (track.m_observations.count(k) == 0) continue;
        obs_num++;
        cid_vec.emplace_back(k);
      }
      if (obs_num < 2) continue;
      id_pt2cams[id] = cid_vec;
    }
    
    for (auto &[id_pt, cam_vec] : id_pt2cams) {
      for (int i = 0; i < cam_vec.size(); ++i) {
        const int id_cam1 = cam_vec[i];
        for (int j = i + 1; j < cam_vec.size(); ++j) {
          const int id_cam2 = cam_vec[j];
          if (covisible[id_cam1].count(id_cam2) == 0) covisible[id_cam1][id_cam2] = 0;
          covisible[id_cam1][id_cam2]++;
        }
      }
    }
    return covisible;
  }

};
