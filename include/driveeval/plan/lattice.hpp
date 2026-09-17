#ifndef DRIVEEVAL_PLAN_LATTICE_HPP
#define DRIVEEVAL_PLAN_LATTICE_HPP

#include <array>
#include <vector>

#include "driveeval/core/trajectory.hpp"
#include "driveeval/core/types.hpp"
#include "driveeval/plan/reference_path.hpp"

namespace drive::plan {

// Jerk-optimal polynomials in the Frenet frame, in the form Werling et al.
// (2010) use: a quintic in lateral offset against time, and a quartic in arc
// length against time because the terminal arc length is free while the
// terminal speed is not.
struct Quintic {
  std::array<Scalar, 6> c{};
  static Quintic fit(Scalar p0, Scalar v0, Scalar a0, Scalar p1, Scalar v1, Scalar a1, Scalar T);
  [[nodiscard]] Scalar at(Scalar t) const;
  [[nodiscard]] Scalar d1(Scalar t) const;
  [[nodiscard]] Scalar d2(Scalar t) const;
  [[nodiscard]] Scalar d3(Scalar t) const;
};

struct Quartic {
  std::array<Scalar, 5> c{};
  static Quartic fit(Scalar p0, Scalar v0, Scalar a0, Scalar v1, Scalar a1, Scalar T);
  [[nodiscard]] Scalar at(Scalar t) const;
  [[nodiscard]] Scalar d1(Scalar t) const;
  [[nodiscard]] Scalar d2(Scalar t) const;
  [[nodiscard]] Scalar d3(Scalar t) const;
};

struct LatticeOptions {
  Scalar horizon{5.0};  // s, trajectory length handed to the cost function
  // Sample spacing, and also the prediction step and the simulator step. Held
  // equal to the dataset's 10 Hz logging rate so that no stage of the pipeline
  // interpolates another stage's output, which removes a whole class of
  // disagreement between what the planner intended and what the metrics scored.
  Scalar dt{0.1};
  // Lateral offsets are sampled relative to the reference path, which already
  // sits on the route's lane centerlines, so 0 is lane keeping.
  std::vector<Scalar> lateral_offsets{-3.2, -1.6, 0.0, 1.6, 3.2};
  // Fractions of the local speed prior. Zero is included so that stopping is
  // always a candidate, which matters at intersections and behind a lead agent.
  std::vector<Scalar> speed_fractions{0.0, 0.3, 0.6, 0.85, 1.0, 1.15};
  // Two terminal times give the sampler both a committed and a hesitant version
  // of each manoeuvre. More than two did not change the chosen trajectory often
  // enough to pay for the candidates.
  std::vector<Scalar> terminal_times{3.0, 5.0};

  [[nodiscard]] std::size_t maxCandidates() const {
    return lateral_offsets.size() * speed_fractions.size() * terminal_times.size();
  }
  [[nodiscard]] std::size_t horizonSteps() const {
    return static_cast<std::size_t>(horizon / dt);
  }
};

struct Candidate {
  Trajectory traj;
  Scalar target_d{0.0};
  Scalar target_v{0.0};
  Scalar terminal_time{0.0};
  Scalar cost{0.0};
  bool feasible{true};
};

// Generates the candidate set into a caller-owned buffer. The buffer is sized
// once by the workspace and reused, so generation performs no allocation.
class Lattice {
 public:
  void reserve(const LatticeOptions& opts);

  // `start` is the ego's current Frenet state plus its current longitudinal
  // acceleration. Returns the number of candidates written.
  std::size_t generate(const ReferencePath& path, const FrenetPoint& start, Scalar a0,
                       const LatticeOptions& opts, const VehicleParams& veh);

  [[nodiscard]] std::size_t size() const { return count_; }
  [[nodiscard]] Candidate& operator[](std::size_t i) { return buf_[i]; }
  [[nodiscard]] const Candidate& operator[](std::size_t i) const { return buf_[i]; }

 private:
  std::vector<Candidate> buf_;
  std::size_t count_{0};
};

}  // namespace drive::plan

#endif  // DRIVEEVAL_PLAN_LATTICE_HPP
