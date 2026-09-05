#include "switchback/plan/planner.hpp"

#include <chrono>
#include <cstdio>
#include <limits>
#include <string>

namespace sb::plan {

std::string PlannerConfig::toJson() const {
  std::string out = "{\"weights\":";
  out += weights.toJson();
  out += ",\"backend\":\"";
  out += toString(backend);
  out += "\",\"prediction\":\"";
  out += predict::toString(prediction);
  out += "\",\"horizon\":";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f", lattice.horizon);
  out += buf;
  out += ",\"dt\":";
  std::snprintf(buf, sizeof(buf), "%.2f", lattice.dt);
  out += buf;
  out += ",\"n_candidates\":";
  out += std::to_string(lattice.maxCandidates());
  out += ",\"corridor_half_width\":";
  std::snprintf(buf, sizeof(buf), "%.2f", corridor_half_width);
  out += buf;
  out += "}";
  return out;
}

void Planner::setup(const PlannerConfig& cfg) {
  cfg_ = cfg;
  const std::size_t steps = cfg_.lattice.horizonSteps();
  lattice_.reserve(cfg_.lattice);
  preds_.reserve(cfg_.max_agents, steps);
  ilqr_.reserve(steps);
  if (cfg_.backend == Backend::kOsqp) qp_.reserve(steps);
  costs_.assign(cfg_.lattice.maxCandidates(), CostResult{});
  out_.refine.traj.pts.reserve(steps + 1);
  fallback_.pts.reserve(steps + 1);
  // Force the refiners' internal result storage to its full size once, so the
  // first real cycle does not pay for a resize.
  out_.refine.traj.pts.resize(steps + 1);
}

void Planner::bind(const sim::World* world, const map::LaneGraph* graph,
                   const map::DrivableArea* area, const ReferencePath* path) {
  world_ = world;
  graph_ = graph;
  area_ = area;
  path_ = path;
}

const PlanOutput& Planner::plan(const EgoState& ego, Scalar ego_accel, std::size_t sim_step) {
  const auto t0 = std::chrono::steady_clock::now();
  out_.ok = false;
  out_.failure = "";
  out_.chosen_index = -1;
  out_.trajectory = nullptr;
  out_.chosen_candidate = nullptr;
  out_.n_candidates = 0;
  out_.n_feasible = 0;
  out_.refine_rejected = false;
  out_.prediction_overflow = false;
  out_.lattice_us = out_.cost_us = out_.refine_us = 0.0;

  if (path_ == nullptr || path_->empty() || world_ == nullptr) {
    out_.failure = "not_bound";
    return out_;
  }

  const FrenetPoint start = path_->toFrenet(Vec2{ego.x, ego.y}, ego.theta, ego.v);

  const auto t1 = std::chrono::steady_clock::now();
  out_.n_candidates = lattice_.generate(*path_, start, ego_accel, cfg_.lattice, cfg_.veh);
  const auto t2 = std::chrono::steady_clock::now();
  out_.lattice_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
  if (out_.n_candidates == 0) {
    out_.failure = "no_candidates";
    return out_;
  }

  preds_.fill(*world_, *graph_, sim_step, cfg_.lattice.horizonSteps(), cfg_.lattice.dt,
              Vec2{ego.x, ego.y}, cfg_.prediction_radius, cfg_.prediction);
  out_.prediction_overflow = preds_.overflowed();

  const auto t3 = std::chrono::steady_clock::now();
  // Choose among feasible candidates first. A feasible candidate is always
  // preferred to an infeasible one regardless of cost, so that a cheap
  // colliding trajectory can never win on arithmetic.
  Scalar best_feasible = std::numeric_limits<Scalar>::max();
  int idx_feasible = -1;
  // Fallback ranking for the case where nothing is feasible. Ranking those by
  // total cost is unstable: every candidate carries the same large collision
  // term, so the choice is decided by the leftovers and flips from cycle to
  // cycle, which showed up as a 39 m/s^3 jerk. Buying the most time before
  // impact is both a defensible rule and a stable one.
  std::size_t best_collision_step = 0;
  Scalar best_fallback_cost = std::numeric_limits<Scalar>::max();
  int idx_any = -1;
  for (std::size_t i = 0; i < out_.n_candidates; ++i) {
    costs_[i] = evaluateCandidate(lattice_[i], *path_, preds_, *area_, cfg_.veh, cfg_.weights,
                                  cfg_.cost_ctx);
    const std::size_t impact =
        costs_[i].collision_agent >= 0 ? costs_[i].collision_step : lattice_[i].traj.pts.size();
    if (impact > best_collision_step ||
        (impact == best_collision_step && costs_[i].total < best_fallback_cost)) {
      best_collision_step = impact;
      best_fallback_cost = costs_[i].total;
      idx_any = static_cast<int>(i);
    }
    if (costs_[i].feasible) {
      ++out_.n_feasible;
      if (costs_[i].total < best_feasible) {
        best_feasible = costs_[i].total;
        idx_feasible = static_cast<int>(i);
      }
    }
  }
  const auto t4 = std::chrono::steady_clock::now();
  out_.cost_us = std::chrono::duration<double, std::micro>(t4 - t3).count();

  out_.chosen_index = idx_feasible >= 0 ? idx_feasible : idx_any;
  if (out_.chosen_index < 0) {
    out_.failure = "no_candidate_scored";
    return out_;
  }
  const auto ci = static_cast<std::size_t>(out_.chosen_index);
  out_.cost = costs_[ci];
  out_.chosen_candidate = &lattice_[ci].traj;
  out_.trajectory = out_.chosen_candidate;
  // No feasible candidate means every option collides or leaves the road. The
  // least-bad one is executed and the cycle is recorded, because a planner that
  // refuses to output anything is not measurable.
  if (idx_feasible < 0) out_.failure = "no_feasible_candidate";
  out_.ok = true;

  if (cfg_.backend == Backend::kNone) {
    out_.plan_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    return out_;
  }

  RefineProblem prob = cfg_.refine;
  prob.reference = out_.chosen_candidate;
  prob.path = path_;
  prob.preds = &preds_;
  prob.x0 = ego;
  prob.dt = cfg_.lattice.dt;
  prob.corridor_half_width = cfg_.corridor_half_width;
  prob.veh = cfg_.veh;

  const auto t5 = std::chrono::steady_clock::now();
  if (cfg_.backend == Backend::kIlqr) {
    ilqr_.solve(prob, out_.refine, cfg_.ilqr);
  } else {
    qp_.solve(prob, out_.refine, cfg_.qp);
  }
  const auto t6 = std::chrono::steady_clock::now();
  out_.refine_us = std::chrono::duration<double, std::micro>(t6 - t5).count();

  // Refinement is only accepted if it is actually better on the shared
  // objective and does not introduce a collision the lattice candidate did not
  // have. An optimiser that converges to a lower tracking cost through a
  // vehicle is worse than the trajectory it started from, and this check is
  // what stops a solver bug from becoming a safety number.
  bool accept = !out_.refine.traj.empty() && out_.refine.cost <= out_.refine.initial_cost;
  if (accept) {
    const bool lattice_clear = out_.cost.collision_agent < 0;
    if (lattice_clear) {
      for (std::size_t k = 0; k < out_.refine.traj.pts.size() && accept; ++k) {
        const TrajPoint& p = out_.refine.traj.pts[k];
        const Box2 ego_box = egoFootprint(p.x, p.y, p.heading, cfg_.veh);
        const std::size_t pk = std::min(k, preds_.horizon());
        for (std::size_t ai = 0; ai < preds_.size(); ++ai) {
          if (!preds_[ai].validAt(pk)) continue;
          if (obbOverlap(ego_box, preds_[ai].boxAt(pk))) {
            accept = false;
            break;
          }
        }
      }
    }
  }
  if (accept) {
    out_.trajectory = &out_.refine.traj;
  } else {
    out_.refine_rejected = true;
  }

  out_.plan_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  return out_;
}

}  // namespace sb::plan
