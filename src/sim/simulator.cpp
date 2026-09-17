#include "driveeval/sim/simulator.hpp"

#include <algorithm>
#include <cmath>

#include "driveeval/core/alloc_count.hpp"
#include "driveeval/sim/tracker.hpp"

namespace drive::sim {

Control extractControl(const Trajectory& traj, Scalar dt, const VehicleParams& veh) {
  Control c;
  if (traj.pts.size() < 2) return c;
  c.a = clampT((traj.pts[1].v - traj.pts[0].v) / dt, veh.max_decel, veh.max_accel);
  // The trajectory's curvature at its first sample is a one-sided difference of
  // the first two headings, which is exactly the turn being commanded now.
  c.delta = clampT(std::atan(traj.pts[0].kappa * veh.wheelbase), -veh.max_steer, veh.max_steer);
  return c;
}

void Simulator::setup(const SimConfig& cfg) {
  cfg_ = cfg;
  planner_.setup(cfg_.planner);
  out_.cycles.reserve(256);
  out_.executed.pts.reserve(256);
  pursuit_.pts.reserve(cfg_.planner.lattice.horizonSteps() + 1);
}

const SimOutput& Simulator::run(const io::ScenarioView& view, io::u32 capabilities) {
  out_.cycles.clear();
  out_.plans.clear();
  out_.executed.pts.clear();
  out_.ok = false;
  out_.failure = "";
  out_.metrics = eval::ScenarioMetrics{};
  out_.metrics.scenario_id = view.id();

  if (!view.hasEgo()) {
    out_.failure = "no_ego";
    out_.metrics.status = "load_fail";
    return out_;
  }

  graph_ = map::LaneGraph::build(view, cfg_.graph_opts);

  const auto ego_track = view.egoTrack();
  std::size_t last_valid = 0;
  for (std::size_t t = 0; t < ego_track.size(); ++t) {
    if (ego_track[t].valid != 0) last_valid = t;
  }

  map::RouteRequest req;
  req.start = Vec2{static_cast<Scalar>(ego_track[0].x), static_cast<Scalar>(ego_track[0].y)};
  req.start_heading = static_cast<Scalar>(ego_track[0].heading);
  // The goal is where the human actually ended up, which is what makes progress
  // measurable against an expert rather than against an arbitrary target.
  req.goal = Vec2{static_cast<Scalar>(ego_track[last_valid].x),
                  static_cast<Scalar>(ego_track[last_valid].y)};
  out_.route = map::findRoute(graph_, req, cfg_.route_weights);
  if (!out_.route.ok()) {
    out_.failure = map::toString(out_.route.status);
    out_.metrics.status = "no_route";
    return out_;
  }

  path_ = plan::ReferencePath::build(graph_, out_.route, cfg_.refpath);
  if (path_.empty()) {
    out_.failure = "empty_reference_path";
    out_.metrics.status = "no_route";
    return out_;
  }

  // The distance field is built only over the route corridor, which is the only
  // region the ego can reach. Building it over the whole map cost 396 ms per
  // scenario against roughly 600 ms of actual planning.
  {
    Vec2 lo{1e18, 1e18};
    Vec2 hi{-1e18, -1e18};
    for (const Vec2& p : path_.points()) {
      lo.x = std::min(lo.x, p.x);
      lo.y = std::min(lo.y, p.y);
      hi.x = std::max(hi.x, p.x);
      hi.y = std::max(hi.y, p.y);
    }
    // The logged ego track too, so the human-baseline replay is covered even
    // where it departs from the planned route.
    for (std::size_t t = 0; t < view.numSteps(); ++t) {
      if (ego_track[t].valid == 0) continue;
      lo.x = std::min(lo.x, static_cast<Scalar>(ego_track[t].x) - 1.0);
      lo.y = std::min(lo.y, static_cast<Scalar>(ego_track[t].y) - 1.0);
      hi.x = std::max(hi.x, static_cast<Scalar>(ego_track[t].x) + 1.0);
      hi.y = std::max(hi.y, static_cast<Scalar>(ego_track[t].y) + 1.0);
    }
    const Scalar pad = 25.0;
    area_ = map::DrivableArea::build(view, Vec2{lo.x - pad, lo.y - pad},
                                     Vec2{hi.x + pad, hi.y + pad});
  }

  world_.reset(view, cfg_.agent_mode, cfg_.reactive, cfg_.perturb);
  if (cfg_.suppress_agent >= 0 &&
      static_cast<std::size_t>(cfg_.suppress_agent) < world_.numAgents()) {
    world_.suppress(static_cast<std::size_t>(cfg_.suppress_agent));
  }
  planner_.bind(&world_, &graph_, &area_, &path_);

  const Scalar dt = view.dt();
  const plan::BicycleModel model{cfg_.planner.veh.wheelbase};
  EgoState ego{static_cast<Scalar>(ego_track[0].x), static_cast<Scalar>(ego_track[0].y),
               static_cast<Scalar>(ego_track[0].heading),
               std::hypot(static_cast<Scalar>(ego_track[0].vx),
                          static_cast<Scalar>(ego_track[0].vy))};
  Scalar ego_accel = 0.0;
  Scalar prev_delta = 0.0;
  Scalar prev_a_cmd = 0.0;
  Scalar prev_a = 0.0;
  Scalar prev_yaw_rate = 0.0;

  std::size_t n_steps = view.numSteps();
  if (cfg_.max_steps > 0) n_steps = std::min(n_steps, cfg_.max_steps);

  std::int64_t alloc_before = 0;
  if (cfg_.count_allocs) alloc_before = allocationCount();
  std::int64_t plan_allocs = 0;

  for (std::size_t step = 0; step + 1 < n_steps; ++step) {
    eval::CycleRecord rec;
    rec.step = step;
    rec.t = static_cast<Scalar>(step) * dt;
    rec.ego = ego;
    rec.a_lon = prev_a;
    rec.yaw_rate = prev_yaw_rate;
    rec.a_lat = ego.v * prev_yaw_rate;
    rec.jerk = 0.0;  // filled below from successive a_lon

    const Box2 ego_box = egoFootprint(ego, cfg_.planner.veh);

    // Exact collision check against the world as it is right now.
    for (std::size_t i = 0; i < world_.numAgents(); ++i) {
      if (i == world_.egoIndex() || !world_.valid(i) || world_.suppressed(i)) continue;
      const io::AgentMeta& meta = world_.meta(i);
      // Static map furniture and background clutter are not collision
      // candidates: Argoverse 2 labels cones and signage as agents, and scoring
      // the ego for clipping a cone would swamp the vehicle collision rate.
      if (meta.type == io::kAgentStatic || meta.type == io::kAgentBackground ||
          meta.type == io::kAgentConstruction) {
        continue;
      }
      if (obbOverlap(ego_box, world_.box(i))) {
        rec.collision_agent = static_cast<int>(i);
        rec.collision_agent_type = static_cast<int>(meta.type);
        rec.ego_struck_from_behind =
            eval::struckFromBehind(ego, cfg_.planner.veh, world_.box(i), world_.vel(i));
        break;
      }
    }

    rec.min_ttc = eval::timeToCollision(ego, cfg_.planner.veh, world_, cfg_.thresholds, dt);
    if ((capabilities & io::kCapDrivableArea) != 0 && !area_.empty()) {
      // The exact scan, not the grid. This runs once per cycle rather than
      // 12,000 times, so its cost is irrelevant, and it keeps the reported
      // off-road metric independent of the cost function's approximation.
      const BoxCorners c = corners(ego_box);
      rec.offroad = 0.0;
      for (const Vec2& q : c) rec.offroad = std::max(rec.offroad, area_.outsideDistanceExact(q));
    }
    // Wrong-way is "aligned with no plausible lane", not "misaligned with the
    // nearest one". Matching the nearest lane at any heading put the ego on the
    // oncoming lane of a two-way road whenever that centerline happened to be
    // marginally closer, which reported 31% of scenarios as wrong-way. The test
    // that means what the metric says is whether ANY lane covering this pose
    // agrees with the ego's heading.
    const auto matches = graph_.matchPose(Vec2{ego.x, ego.y}, ego.theta, 5.0, kPi, true);
    if (!matches.empty()) {
      rec.lane_matched = true;
      Scalar best_err = kPi;
      for (const auto& mm : matches) best_err = std::min(best_err, std::abs(mm.heading_error));
      rec.heading_error = best_err;
    }
    const FrenetPoint f = path_.toFrenet(Vec2{ego.x, ego.y}, ego.theta, ego.v);
    rec.s = f.s;
    rec.d = f.d;
    rec.speed_prior = path_.speedPriorAt(f.s);

    Control ctrl;
    if (cfg_.replay_logged_ego) {
      // The ego is placed exactly where the log says. This produces the human's
      // own score under the identical metric suite, which is the only honest
      // floor to read the planner's rates against.
      rec.plan_us = 0.0;
      ctrl = Control{};
    } else if (cfg_.pure_pursuit_only) {
      // Baseline planner: track the reference path geometrically and hold the
      // local speed prior. No obstacle reasoning at all, which is stated
      // plainly wherever its numbers are reported.
      const Scalar lookahead = cfg_.pure_pursuit_lookahead_base +
                               cfg_.pure_pursuit_lookahead_gain * ego.v;
      const Pose2 aim = path_.poseAt(f.s + lookahead);
      const Scalar desired = (Vec2{aim.x, aim.y} - Vec2{ego.x, ego.y}).angle();
      ctrl.delta = clampT(angleDiff(desired, ego.theta), -cfg_.planner.veh.max_steer,
                          cfg_.planner.veh.max_steer);
      ctrl.a = clampT((rec.speed_prior - ego.v) / 1.0, cfg_.planner.veh.max_decel,
                      cfg_.planner.veh.max_accel);
      rec.plan_us = 0.0;
    } else {
      drive::AllocScope scope;
      const plan::PlanOutput& po = planner_.plan(ego, ego_accel, step);
      if (cfg_.count_allocs) plan_allocs += scope.delta();
      rec.plan_us = po.plan_us;
      rec.refine_us = po.refine_us;
      rec.refine_iters = po.refine.iterations;
      rec.refine_converged = po.refine.converged;
      rec.no_feasible_candidate = (po.n_feasible == 0);
      rec.min_margin = po.cost.min_margin;
      if (!po.ok || po.trajectory == nullptr) {
        out_.failure = po.failure;
        out_.metrics.status = "planner_fail";
        out_.cycles.push_back(rec);
        return out_;
      }
      ctrl = trackTrajectory(*po.trajectory, ego, cfg_.planner.veh, cfg_.tracker);

      if (cfg_.dump_plan_stride > 0 && step % cfg_.dump_plan_stride == 0) {
        PlanSnapshot snap;
        snap.t = rec.t;
        snap.chosen = po.chosen_index;
        snap.refine_used = (po.trajectory == &po.refine.traj);
        const plan::Lattice& lat = planner_.lattice();
        snap.candidates.reserve(po.n_candidates);
        for (std::size_t ci = 0; ci < po.n_candidates; ++ci) {
          snap.candidates.push_back(lat[ci].traj);
          snap.costs.push_back(planner_.candidateCosts()[ci].total);
          snap.feasible.push_back(planner_.candidateCosts()[ci].feasible ? char{1} : char{0});
        }
        if (snap.refine_used) snap.refined = po.refine.traj;
        out_.plans.push_back(std::move(snap));
      }
    }

    // Steering rate limit. Without it the planner can command a full lock
    // reversal between consecutive cycles, which no steering actuator can do
    // and which would make the comfort metrics measure the planner's
    // discretisation rather than its behaviour.
    const Scalar max_step = cfg_.planner.veh.max_steer_rate * dt;
    ctrl.delta = clampT(ctrl.delta, prev_delta - max_step, prev_delta + max_step);
    prev_delta = ctrl.delta;

    // Acceleration slew limit, for the same reason. Without it the speed
    // controller saturated at the acceleration limit and changed sign between
    // consecutive cycles, which put median jerk at 14 m/s^3 against a comfort
    // threshold of 8.37 and made 79% of scenarios report a comfort violation
    // that was an artefact of the controller rather than of the plan.
    const Scalar max_da = cfg_.tracker.max_jerk * dt;
    ctrl.a = clampT(ctrl.a, prev_a_cmd - max_da, prev_a_cmd + max_da);
    prev_a_cmd = ctrl.a;

    out_.cycles.push_back(rec);

    EgoState next = model.step(ego, ctrl, dt);
    if (cfg_.replay_logged_ego) {
      const std::size_t ei = world_.egoIndex();
      if (world_.loggedValid(ei, step + 1)) {
        const Vec2 p = world_.loggedPos(ei, step + 1);
        // Speed from the position difference, not from the logged velocity
        // field. Argoverse 2's per-timestep velocities are themselves estimates,
        // and differencing them at 10 Hz produced a median longitudinal
        // acceleration of 26 m/s^2 and a median jerk of 141 m/s^3 for a human
        // driving normally, which says the field is too noisy to differentiate
        // rather than that the driver was violent. Positions difference far
        // more cleanly, and this is the same quantity the planner is scored on.
        const Scalar v =
            world_.loggedValid(ei, step) ? distance(p, world_.loggedPos(ei, step)) / dt : ego.v;
        next = EgoState{p.x, p.y, world_.loggedHeading(ei, step + 1), v};
      }
    }
    prev_yaw_rate = angleDiff(next.theta, ego.theta) / dt;
    prev_a = (next.v - ego.v) / dt;
    ego_accel = prev_a;
    ego = next;
    ego.v = std::max(Scalar{0.0}, ego.v);

    world_.step(step, egoFootprint(ego, cfg_.planner.veh), ego, dt);
  }

  // Jerk from successive longitudinal accelerations, after the loop so it uses
  // the same differencing the trajectory type does.
  for (std::size_t i = 0; i < out_.cycles.size(); ++i) {
    const std::size_t lo = i > 0 ? i - 1 : 0;
    const std::size_t hi = i + 1 < out_.cycles.size() ? i + 1 : out_.cycles.size() - 1;
    const Scalar span = (static_cast<Scalar>(hi) - static_cast<Scalar>(lo)) * dt;
    out_.cycles[i].jerk =
        span > kEps ? (out_.cycles[hi].a_lon - out_.cycles[lo].a_lon) / span : 0.0;
  }

  out_.executed.pts.resize(out_.cycles.size());
  for (std::size_t i = 0; i < out_.cycles.size(); ++i) {
    TrajPoint& p = out_.executed.pts[i];
    const eval::CycleRecord& c = out_.cycles[i];
    p.t = c.t;
    p.x = c.ego.x;
    p.y = c.ego.y;
    p.heading = c.ego.theta;
    p.v = c.ego.v;
    p.s = c.s;
    p.d = c.d;
  }

  eval::MetricInputs in;
  in.cycles = &out_.cycles;
  in.world = &world_;
  in.path = &path_;
  in.area = &area_;
  in.dt = dt;
  in.capabilities = capabilities;
  in.kinematics_measurable = !cfg_.replay_logged_ego;
  out_.metrics = eval::computeMetrics(in, cfg_.thresholds);
  out_.metrics.scenario_id = view.id();
  out_.metrics.hot_path_allocs = cfg_.count_allocs ? plan_allocs : -1;
  if (cfg_.count_allocs) {
    // Scenario-level setup allocates; only the per-cycle delta is the claim.
    (void)alloc_before;
  }
  out_.ok = true;
  return out_;
}

}  // namespace drive::sim
