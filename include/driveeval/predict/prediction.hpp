#ifndef DRIVEEVAL_PREDICT_PREDICTION_HPP
#define DRIVEEVAL_PREDICT_PREDICTION_HPP

#include <cstddef>
#include <vector>

#include "driveeval/core/geometry.hpp"
#include "driveeval/core/types.hpp"
#include "driveeval/io/cache_reader.hpp"
#include "driveeval/map/lane_graph.hpp"
#include "driveeval/sim/world.hpp"

namespace drive::predict {

enum class Model {
  // What the planner actually uses. Every agent holds its current velocity.
  kConstantVelocity,
  // Agents follow their matched lane at their current speed, which is better in
  // curves and is the more common production baseline.
  kLaneFollow,
  // Reads the agent's logged future. Not a prediction -- it is an oracle, used
  // only by the ablation tool to separate planning failures from prediction
  // failures. Never used in a reported metric run.
  kLoggedOracle,
};

const char* toString(Model m);

struct AgentPrediction {
  std::size_t agent_index{0};
  io::u32 type{0};
  Scalar length{0.0};
  Scalar width{0.0};
  // Indexed by horizon step. Sized to horizon + 1 and never reallocated after
  // the workspace is reserved.
  std::vector<Vec2> pos;
  std::vector<Scalar> heading;
  std::vector<Scalar> speed;
  std::vector<char> valid;
  // Axis-aligned bound of this agent's whole predicted path over the horizon,
  // already grown by its own circumradius. Computed once per cycle so the cost
  // function can reject an agent for a whole candidate with one box test
  // instead of one circle test per trajectory sample.
  Vec2 aabb_min{};
  Vec2 aabb_max{};
  // Three-disc cover of this agent's footprint at each horizon step, laid out
  // as [step * 3 + i]. Precomputed once per cycle and reused by all sixty
  // candidates, which is the difference between 3 x 60 x 51 disc placements and
  // 3 x 51 of them.
  std::vector<Vec2> discs;
  Scalar disc_radius{0.0};

  [[nodiscard]] Box2 boxAt(std::size_t k) const {
    return Box2{pos[k], heading[k], length, width};
  }
  [[nodiscard]] bool validAt(std::size_t k) const { return valid[k] != 0; }
};

// Predictions for every agent that matters this cycle. Owned by the planner
// workspace and refilled in place, so a cycle performs no allocation.
class PredictionSet {
 public:
  void reserve(std::size_t max_agents, std::size_t horizon_steps);

  // Fill from the world's *current* agent states, predicting forward `horizon`
  // steps at `dt`. Reading the world rather than the cache is what makes the
  // planner see reactive agents where they actually are instead of where they
  // were recorded. Agents farther than `radius` from `ego` are skipped, which
  // is the largest constant-factor win in the cost evaluation.
  void fill(const sim::World& world, const map::LaneGraph& graph, std::size_t step,
            std::size_t horizon, Scalar dt, const Vec2& ego, Scalar radius, Model model);

  [[nodiscard]] std::size_t size() const { return count_; }
  [[nodiscard]] const AgentPrediction& operator[](std::size_t i) const { return agents_[i]; }
  [[nodiscard]] std::size_t horizon() const { return horizon_; }
  [[nodiscard]] std::size_t capacity() const { return agents_.size(); }
  [[nodiscard]] bool overflowed() const { return overflowed_; }

 private:
  std::vector<AgentPrediction> agents_;
  std::size_t count_{0};
  std::size_t horizon_{0};
  bool overflowed_{false};
};

}  // namespace drive::predict

#endif  // DRIVEEVAL_PREDICT_PREDICTION_HPP
