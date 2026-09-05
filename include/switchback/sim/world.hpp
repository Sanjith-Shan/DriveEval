#ifndef SWITCHBACK_SIM_WORLD_HPP
#define SWITCHBACK_SIM_WORLD_HPP

#include <cstddef>
#include <vector>

#include "switchback/core/geometry.hpp"
#include "switchback/core/types.hpp"
#include "switchback/io/cache_reader.hpp"

namespace sb::sim {

enum class AgentMode {
  // Other agents follow their recorded tracks and ignore the ego completely.
  // This is the mode almost every student project evaluates in, and the gap
  // between it and kReactive is the project's headline measurement.
  kLogReplay,
  // Other agents run IDM longitudinally against whichever obstacle is ahead of
  // them, the ego included, and pure-pursuit laterally along their own recorded
  // path. Their free-flow target speed at each step is their recorded speed, so
  // an unobstructed agent reproduces its log and an obstructed one yields.
  kReactive,
};

const char* toString(AgentMode m);
AgentMode agentModeFromString(const std::string& s);

struct ReactiveParams {
  // Intelligent Driver Model. Values are the commonly cited urban set; the
  // point of the harness is not that these are calibrated but that the
  // sensitivity of the headline to them can be measured, which sb_sweep does.
  Scalar a_max{1.5};
  Scalar b_comfort{2.0};
  Scalar min_gap{2.0};       // s0
  Scalar headway{1.5};       // T
  Scalar accel_exponent{4.0};  // delta
  Scalar max_decel{-6.0};

  // Pure pursuit along the agent's own logged path.
  Scalar lookahead_base{3.0};
  Scalar lookahead_gain{0.6};

  // Leader detection, in the follower's local frame.
  Scalar leader_window{60.0};
  Scalar leader_lateral_tol{2.2};
};

// A controlled change to one agent, for the criticality sweeps. Shifting a
// single agent's track in time is the cleanest one-dimensional perturbation of
// an interaction: it changes when the ego meets that agent without changing
// where either of them drives, so a failure boundary found by sweeping it is a
// statement about timing rather than about geometry.
struct Perturbation {
  int agent{-1};
  int time_shift{0};         // steps, positive delays the agent
  Scalar speed_scale{1.0};   // applied to the agent's reactive free-flow target

  [[nodiscard]] bool active() const {
    return agent >= 0 && (time_shift != 0 || std::abs(speed_scale - 1.0) > 1e-9);
  }
};

// The set of agents at the current simulation step, under whichever agent model
// is active. Everything downstream -- prediction, collision checking, metrics --
// reads the world through this interface, so log-replay and reactive differ in
// exactly one place instead of being threaded through the whole harness.
class World {
 public:
  void reset(const io::ScenarioView& view, AgentMode mode, const ReactiveParams& rp,
             const Perturbation& perturb = {});

  // Advance non-ego agents from `step` to `step + 1`. The ego's footprint is
  // passed in because in reactive mode agents must be able to see it.
  void step(std::size_t step, const Box2& ego_box, const EgoState& ego, Scalar dt);

  [[nodiscard]] AgentMode mode() const { return mode_; }
  [[nodiscard]] std::size_t numAgents() const { return n_; }
  [[nodiscard]] std::size_t egoIndex() const { return ego_index_; }
  [[nodiscard]] const io::AgentMeta& meta(std::size_t i) const { return view_->agents[i]; }

  [[nodiscard]] bool valid(std::size_t i) const { return cur_[i].valid; }
  [[nodiscard]] Vec2 pos(std::size_t i) const { return cur_[i].pos; }
  [[nodiscard]] Scalar heading(std::size_t i) const { return cur_[i].heading; }
  [[nodiscard]] Scalar speed(std::size_t i) const { return cur_[i].speed; }
  [[nodiscard]] Vec2 vel(std::size_t i) const {
    return Vec2{std::cos(cur_[i].heading), std::sin(cur_[i].heading)} * cur_[i].speed;
  }
  [[nodiscard]] Box2 box(std::size_t i) const {
    return Box2{cur_[i].pos, cur_[i].heading, static_cast<Scalar>(view_->agents[i].length),
                static_cast<Scalar>(view_->agents[i].width)};
  }

  // The recorded track, independent of the agent model. Used by the oracle
  // predictor and by the reactive-model validation check.
  [[nodiscard]] bool loggedValid(std::size_t i, std::size_t step) const;
  [[nodiscard]] Vec2 loggedPos(std::size_t i, std::size_t step) const;
  [[nodiscard]] Scalar loggedHeading(std::size_t i, std::size_t step) const;
  [[nodiscard]] Scalar loggedSpeed(std::size_t i, std::size_t step) const;

  [[nodiscard]] const io::ScenarioView& view() const { return *view_; }
  // Set when an agent is excluded, for the agent-removal ablation.
  void suppress(std::size_t i) { cur_[i].suppressed = true; }
  [[nodiscard]] bool suppressed(std::size_t i) const { return cur_[i].suppressed; }

 private:
  struct AgentState {
    Vec2 pos{};
    Scalar heading{0.0};
    Scalar speed{0.0};
    Scalar s{0.0};  // arc length along its own logged path
    bool valid{false};
    bool suppressed{false};
  };
  // Each reactive agent follows the path it actually drove. Stored as a plain
  // polyline rather than a full ReferencePath because pure pursuit needs only
  // position lookup, and there is one of these per agent per scenario.
  struct AgentPath {
    std::vector<Vec2> pts;
    std::vector<Scalar> cum_s;
    std::vector<Scalar> speed;  // logged speed at each path sample
    [[nodiscard]] bool usable() const { return pts.size() >= 2; }
  };

  [[nodiscard]] Scalar idmAccel(std::size_t i, Scalar gap, Scalar lead_speed,
                                Scalar target_speed) const;
  [[nodiscard]] Scalar leaderGap(std::size_t i, const Box2& ego_box, Scalar ego_speed,
                                 Scalar& lead_speed) const;

  // Track index for agent i at simulation step t, after any perturbation.
  [[nodiscard]] long shifted(std::size_t i, std::size_t step) const;

  const io::ScenarioView* view_{nullptr};
  AgentMode mode_{AgentMode::kLogReplay};
  Perturbation perturb_;
  ReactiveParams rp_;
  std::size_t n_{0};
  std::size_t ego_index_{0};
  std::vector<AgentState> cur_;
  std::vector<AgentPath> paths_;
};

}  // namespace sb::sim

#endif  // SWITCHBACK_SIM_WORLD_HPP
