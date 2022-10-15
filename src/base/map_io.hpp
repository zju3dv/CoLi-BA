//
// Created by SENSETIME\yezhichao1 on 2020/4/5.
//

#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "base/map.h"

// for mmap:
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// NOTATION: compatible with uint64

using Clock = std::chrono::high_resolution_clock;

inline double tloop(const Clock::time_point &t_start, const Clock::time_point &t_end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
}

struct membuf : std::streambuf {
  membuf(char *begin, char *end) { this->setg(begin, begin, end); }
};

inline void handle_error(const char *msg) {
  perror(msg);
  exit(255);
}

inline std::string read_binary_file_to_string(const std::string &path) {
  std::string buffer;
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cerr << "Failed to open " << path << "\n";
    handle_error("open");
  }
  // obtain file size
  struct stat sb;
  if (fstat(fd, &sb) == -1) handle_error("fstat");
  size_t length = sb.st_size;
  const char *addr = static_cast<const char *>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));
  if (addr == MAP_FAILED) handle_error("mmap");
  buffer.resize(length);
  std::memcpy((void *)buffer.data(), addr, length);
  close(fd);
  return buffer;
}

template <typename T>
inline void read_data(std::istream &file, T &data, bool log = false) {
  file.read(reinterpret_cast<char *>(&data), sizeof(T));
  if (log) std::cout << data << std::endl;
}

template <typename T>
inline T read_data2(std::istream &file, bool log = false) {
  T data;
  file.read(reinterpret_cast<char *>(&data), sizeof(T));
  if (log) std::cout << data << std::endl;
  return data;
}

template <typename T>
inline void read_data_vec(std::istream &file, T *data, int num, bool log = false) {
  file.read(reinterpret_cast<char *>(data), num * sizeof(T));
  if (log) {
    for (int i = 0; i < num; ++i) std::cout << data[i] << " ";
    std::cout << std::endl;
  }
}

template <typename T>
inline void read_buf(char *&buf, T *data, int num = 1, bool log = false) {
  std::memcpy((void *)data, buf, num * sizeof(T));
  buf += num * sizeof(T);
  if (log) std::cout << *data << std::endl;
}

inline void read_image_name(std::istream &file, std::string &name, bool log = false) {
  std::getline(file, name, '\0');
  if (log) std::cout << name << std::endl;
}

inline void read_cameras(const std::string &file_name, Map &map) {
  // TODO set more camera model
  std::string str_buf = std::move(read_binary_file_to_string(file_name));
  membuf sbuf(str_buf.data(), str_buf.data() + str_buf.length());
  std::istream file(&sbuf);
  int64_t camera_num;
  int camera_id, camera_model_id;
  int64_t width, height;
  double param[4];
  read_data(file, camera_num);
  map.m_camera_models.clear();
  for (int i = 0; i < camera_num; ++i) {
    read_data(file, camera_id);
    read_data(file, camera_model_id);
    read_data(file, width);
    read_data(file, height);
    read_data_vec(file, param, 4);
    CameraModel camera_model;
    camera_model.id = camera_id;
    camera_model.camera_params = {param[0], param[1], param[2], param[3]};
    map.m_camera_models[camera_id] = camera_model;
    // camera_model.log();
  }
}

inline void read_images(const std::string &file_name, Map &map) {
  std::string str_buf = std::move(read_binary_file_to_string(file_name));
  membuf sbuf(str_buf.data(), str_buf.data() + str_buf.length());
  std::istream file(&sbuf);

  map.m_frames.clear();
  auto num_images = read_data2<int64_t>(file);

  for (int i = 0; i < num_images; ++i) {
    Frame t_frame;
    read_data(file, t_frame.id);
    read_data_vec(file, t_frame.Tcw.q.data(), 4);
    read_data_vec(file, t_frame.Tcw.t.data(), 3);
    auto camera_id = read_data2<int>(file);
    t_frame.m_camera_model = &(map.m_camera_models[camera_id]);
    read_image_name(file, t_frame.name);
    auto num_points = read_data2<int64_t>(file);
    t_frame.m_points.resize(num_points);
    t_frame.m_track_ids.resize(num_points);

    {  // read point block
      int buf_size = num_points * (8 * 3);
      char buf[buf_size];
      read_data_vec(file, buf, buf_size);
      char *ptr = buf;
      int64_t track_id;
      for (int k = 0; k < num_points; ++k) {
        read_buf(ptr, t_frame.m_points[k].data(), 2);
        read_buf(ptr, &track_id);
        t_frame.m_track_ids[k] = track_id;
      }
    }
    t_frame.registered = true;
    map.m_frames[t_frame.id] = t_frame;
  }
}

inline void read_points3d(const std::string &file_name, Map &map) {
  std::string str_buf = std::move(read_binary_file_to_string(file_name));
  membuf sbuf(str_buf.data(), str_buf.data() + str_buf.length());
  std::istream file(&sbuf);

  map.m_tracks.clear();
  int count_multi_obs = 0;
  auto num_points3D = read_data2<int64_t>(file);
  for (size_t i = 0; i < num_points3D; ++i) {
    auto point3D_id = read_data2<int64_t>(file);
    if (file.eof()) break;
    Track t_track; 
    t_track.id = point3D_id;
    CHECK_EQ(t_track.id,point3D_id);
    read_data_vec(file, t_track.m_point3d.data(), 3);
    read_data_vec(file, t_track.color, 3);
    read_data(file, t_track.error);
    auto num_obs = read_data2<int64_t>(file);
    t_track.m_observations.clear();
    for (size_t k = 0; k < num_obs; ++k) {
      int image_id, point2D_idx;
      read_data(file, image_id);
      read_data(file, point2D_idx);
      // there may are multiple 2d point correspondence in a image
      if (t_track.m_observations.count(image_id) == 0) {
        t_track.m_observations[image_id] = point2D_idx;
      } else {
        // std::cout<<"multi 2d point for a 3d point\n";
        // std::cout<<t_track.id<<" "<<image_id<<" "<<t_track.m_observations[image_id]<<" "<<point2D_idx<<"\n";
        count_multi_obs++;
      }
    }
    if (file.eof()) break;
    assert(map.m_tracks.count(t_track.id) == 0);
    map.m_tracks[t_track.id] = t_track;
    // std::cout<<point3D_id<<" "<<num_points3D<<std::endl;
    // the data of ECIM is not good
    // if (point3D_id == num_points3D - 1) break;
  }
  // std::cout<<count_multi_obs<<std::endl;
}

inline void read_map(std::string path, Map &map) {
  read_cameras(path + "cameras.bin", map);
  read_images(path + "images.bin", map);
  read_points3d(path + "points3D.bin", map);

  map.normalized_rotation();
  map.set_normalized_points();
}