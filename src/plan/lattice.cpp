#include "driveeval/plan/lattice.hpp"

#include <algorithm>
#include <cmath>

namespace drive::plan {

Quintic Quintic::fit(Scalar p0, Scalar v0, Scalar a0, Scalar p1, Scalar v1, Scalar a1, Scalar T) {
  Quintic q;
  q.c[0] = p0;
  q.c[1] = v0;
  q.c[2] = 0.5 * a0;
  if (T < kEps) return q;
  const Scalar T2 = T * T;
  const Scalar T3 = T2 * T;
  const Scalar T4 = T3 * T;
  const Scalar T5 = T4 * T;
  // Right-hand side after removing the contribution of the first three terms.
  const Scalar b0 = p1 - (q.c[0] + q.c[1] * T + q.c[2] * T2);
  const Scalar b1 = v1 - (q.c[1] + 2.0 * q.c[2] * T);
  const Scalar b2 = a1 - 2.0 * q.c[2];
  // Closed-form inverse of the 3x3 system in (c3, c4, c5). Written out rather
  // than solved numerically because this runs once per candidate per cycle.
  q.c[3] = (10.0 * b0 - 4.0 * b1 * T + 0.5 * b2 * T2) / T3;
  q.c[4] = (-15.0 * b0 + 7.0 * b1 * T - 1.0 * b2 * T2) / T4;
  q.c[5] = (6.0 * b0 - 3.0 * b1 * T + 0.5 * b2 * T2) / T5;
  return q;
}

Scalar Quintic::at(Scalar t) const {
  return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
}
Scalar Quintic::d1(Scalar t) const {
  return c[1] + t * (2.0 * c[2] + t * (3.0 * c[3] + t * (4.0 * c[4] + t * 5.0 * c[5])));
}
Scalar Quintic::d2(Scalar t) const {
  return 2.0 * c[2] + t * (6.0 * c[3] + t * (12.0 * c[4] + t * 20.0 * c[5]));
}
Scalar Quintic::d3(Scalar t) const { return 6.0 * c[3] + t * (24.0 * c[4] + t * 60.0 * c[5]); }

Quartic Quartic::fit(Scalar p0, Scalar v0, Scalar a0, Scalar v1, Scalar a1, Scalar T) {
  Quartic q;
  q.c[0] = p0;
  q.c[1] = v0;
  q.c[2] = 0.5 * a0;
  if (T < kEps) return q;
  const Scalar T2 = T * T;
  const Scalar T3 = T2 * T;
  const Scalar b1 = v1 - (q.c[1] + 2.0 * q.c[2] * T);
  const Scalar b2 = a1 - 2.0 * q.c[2];
  // Terminal position is free, so only the velocity and acceleration
  // conditions remain and the system is 2x2 in (c3, c4).
  q.c[3] = (3.0 * b1 - b2 * T) / (3.0 * T2);
  q.c[4] = (-2.0 * b1 + b2 * T) / (4.0 * T3);
  return q;
}

Scalar Quartic::at(Scalar t) const {
  return c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])));
}
Scalar Quartic::d1(Scalar t) const {
  return c[1] + t * (2.0 * c[2] + t * (3.0 * c[3] + t * 4.0 * c[4]));
}
Scalar Quartic::d2(Scalar t) const { return 2.0 * c[2] + t * (6.0 * c[3] + t * 12.0 * c[4]); }
Scalar Quartic::d3(Scalar t) const { return 6.0 * c[3] + t * 24.0 * c[4]; }

void Lattice::reserve(const LatticeOptions& opts) {
  const std::size_t n = opts.maxCandidates();
  const std::size_t len = opts.horizonSteps() + 1;
  buf_.resize(n);
  for (Candidate& c : buf_) {
    c.traj.pts.reserve(len);
    c.traj.pts.resize(len);
  }
  count_ = 0;
}

std::size_t Lattice::generate(const ReferencePath& path, const FrenetPoint& start, Scalar a0,
                              const LatticeOptions& opts, const VehicleParams& veh) {
  count_ = 0;
  if (path.empty()) return 0;
  const std::size_t steps = opts.horizonSteps();
  const Scalar kappa_max = veh.maxCurvature();

  for (const Scalar T : opts.terminal_times) {
    if (T < kEps) continue;
    // The lateral polynomial ends with zero lateral velocity and acceleration,
    // so every candidate finishes settled in its target offset rather than
    // still drifting across it.
    for (const Scalar d1 : opts.lateral_offsets) {
      const Quintic lat = Quintic::fit(start.d, start.dd, 0.0, d1, 0.0, 0.0, T);
      for (const Scalar frac : opts.speed_fractions) {
        if (count_ >= buf_.size()) return count_;
        const Scalar v_prior = path.speedPriorAt(start.s);
        const Scalar v1 = clampT(frac * v_prior, Scalar{0.0}, veh.max_speed);
        // The measured initial acceleration is clamped before it becomes a
        // boundary condition. Honouring a hard -6 m/s^2 exactly forces the
        // quartic to undershoot through zero velocity within a few tenths of a
        // second for every terminal speed at once, which left the sampler with
        // no candidates at all in exactly the situations -- hard braking -- where
        // it most needs them.
        const Scalar a0c = clampT(a0, Scalar{-4.0}, Scalar{3.0});
        const Quartic lon = Quartic::fit(start.s, std::max(Scalar{0.0}, start.ds), a0c, v1, 0.0, T);

        Candidate& cand = buf_[count_];
        cand.target_d = d1;
        cand.target_v = v1;
        cand.terminal_time = T;
        cand.feasible = true;
        cand.cost = 0.0;
        if (cand.traj.pts.size() != steps + 1) cand.traj.pts.resize(steps + 1);

        // A stopping candidate is truncated at the stop, not discarded. The
        // quartic keeps curving after its velocity reaches zero, and following
        // it would drive the ego backwards; holding position instead is what
        // the candidate actually means.
        bool stopped = false;
        Scalar s_stop = 0.0;
        for (std::size_t k = 0; k <= steps; ++k) {
          const Scalar t = static_cast<Scalar>(k) * opts.dt;
          // Beyond the terminal time the polynomial is no longer valid, so the
          // candidate coasts at its terminal speed along the reference instead
          // of extrapolating a quartic that will diverge.
          const Scalar tc = std::min(t, T);
          Scalar s = lon.at(tc);
          Scalar ds = lon.d1(tc);
          if (t > T) s += ds * (t - T);
          if (!stopped && ds < 0.0) {
            stopped = true;
            s_stop = s;
          }
          if (stopped) {
            s = s_stop;
            ds = 0.0;
          }
          const Scalar d = lat.at(tc);
          const Scalar dd = lat.d1(tc);
          s = clampT(s, Scalar{0.0}, path.length());

          TrajPoint& p = cand.traj.pts[k];
          const Vec2 xy = path.toCartesian(s, d);
          p.t = t;
          p.x = xy.x;
          p.y = xy.y;
          p.s = s;
          p.d = d;
          // Heading combines the reference tangent with the lateral drift,
          // which is the standard Frenet-to-Cartesian heading and is exact for
          // small curvature times offset.
          const Scalar tangent = path.poseAt(s).theta;
          p.heading = wrapAngle(tangent + std::atan2(dd, std::max(ds, Scalar{0.1})));
          p.v = std::hypot(ds, dd);
        }

        cand.traj.differentiate();
        // Feasibility is a property of the candidate, not a cost. An
        // infeasible candidate stays in the set so the report can say how many
        // were rejected and why, but it can never be chosen.
        for (const TrajPoint& p : cand.traj.pts) {
          if (std::abs(p.kappa) > kappa_max * 1.25 || p.v > veh.max_speed * 1.1 ||
              p.a > veh.max_accel * 1.5 || p.a < veh.max_decel * 1.5) {
            cand.feasible = false;
            break;
          }
        }
        ++count_;
      }
    }
  }
  return count_;
}

}  // namespace drive::plan
