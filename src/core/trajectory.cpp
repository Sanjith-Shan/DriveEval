#include "driveeval/core/trajectory.hpp"

#include <cmath>

namespace drive {

void Trajectory::differentiate() {
  const std::size_t n = pts.size();
  if (n < 2) {
    if (n == 1) {
      pts[0].a = 0.0;
      pts[0].jerk = 0.0;
      pts[0].kappa = 0.0;
      pts[0].yaw_rate = 0.0;
      pts[0].a_lat = 0.0;
    }
    return;
  }

  // Speed and heading come from the geometry, so that a trajectory built by any
  // producer -- lattice, optimiser, or the simulator's executed path -- is
  // differentiated identically.
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const Scalar dt = pts[i + 1].t - pts[i].t;
    if (dt > kEps) {
      const Scalar dx = pts[i + 1].x - pts[i].x;
      const Scalar dy = pts[i + 1].y - pts[i].y;
      const Scalar ds = std::hypot(dx, dy);
      // v is taken as signed along the current heading, which keeps a reversing
      // sample from reading as forward motion.
      const Scalar along = dx * std::cos(pts[i].heading) + dy * std::sin(pts[i].heading);
      pts[i].v = (along >= 0.0 ? ds : -ds) / dt;
    }
  }
  pts[n - 1].v = pts[n - 2].v;

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = i > 0 ? i - 1 : 0;
    const std::size_t hi = i + 1 < n ? i + 1 : n - 1;
    const Scalar dt = pts[hi].t - pts[lo].t;
    if (dt > kEps) {
      pts[i].a = (pts[hi].v - pts[lo].v) / dt;
      pts[i].yaw_rate = angleDiff(pts[hi].heading, pts[lo].heading) / dt;
    } else {
      pts[i].a = 0.0;
      pts[i].yaw_rate = 0.0;
    }
    pts[i].kappa = std::abs(pts[i].v) > 0.1 ? pts[i].yaw_rate / pts[i].v : 0.0;
    pts[i].a_lat = pts[i].v * pts[i].yaw_rate;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t lo = i > 0 ? i - 1 : 0;
    const std::size_t hi = i + 1 < n ? i + 1 : n - 1;
    const Scalar dt = pts[hi].t - pts[lo].t;
    pts[i].jerk = dt > kEps ? (pts[hi].a - pts[lo].a) / dt : 0.0;
  }
}

}  // namespace drive
