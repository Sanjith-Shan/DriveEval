#ifndef SWITCHBACK_SIM_TRACKER_HPP
#define SWITCHBACK_SIM_TRACKER_HPP

#include "switchback/core/trajectory.hpp"
#include "switchback/core/types.hpp"

namespace sb::sim {

struct TrackerParams {
  Scalar lookahead_base{3.0};
  Scalar lookahead_gain{0.55};  // seconds of preview
  Scalar lookahead_min{2.5};
  Scalar lookahead_max{18.0};
  Scalar speed_preview{0.6};  // s ahead on the plan to read the target speed
  Scalar speed_tau{0.5};      // s, first-order lag on closing the speed error
  Scalar heading_gain{0.6};   // extra steering per radian of heading error
  // Limit on how fast the commanded acceleration may change, m/s^3. Set just
  // under nuPlan's 8.37 m/s^3 comfort bound so that the controller cannot be
  // the thing that breaches it.
  Scalar max_jerk{6.0};
};

// Follow a planned trajectory from the ego's *actual* pose.
//
// The planner replans every cycle, so it is tempting to treat the first control
// of the plan as the command. That is pure feedforward: the plan is generated
// in a Frenet frame from the ego's current arc length, so its first sample
// carries the planned heading rather than the ego's, and nothing in the loop
// ever corrects the difference. Measured on Argoverse scenario
// 00010486-9a07-48ae-b493-cf4545855937, that let lateral error grow from 0.02 m
// to 3.3 m through one intersection turn, which put the ego in the neighbouring
// lane, made every lattice candidate collide with logged traffic, and stopped
// it dead. Pure pursuit against the plan closes that loop.
Control trackTrajectory(const Trajectory& plan, const EgoState& ego, const VehicleParams& veh,
                        const TrackerParams& tp);

}  // namespace sb::sim

#endif  // SWITCHBACK_SIM_TRACKER_HPP
