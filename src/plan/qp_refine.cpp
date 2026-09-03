// OSQP refinement backend.
//
// Sequential quadratic programming on the same problem iLQR solves. Each pass
// linearises the bicycle dynamics and the obstacle barrier about the current
// iterate and solves one sparse QP in the deviation variables.
//
// The decision vector is
//   z = [dx_0 .. dx_N, du_0 .. du_{N-1}],  dx in R^4, du in R^2
// so the horizon length fixes the problem dimensions. Because the planner's
// horizon never changes at run time, the sparsity pattern is built once in
// reserve() and every cycle only refills the numeric values and warm-starts
// from the previous solution. Rebuilding the factorisation each cycle instead
// would have made the head-to-head against iLQR a comparison of setup cost
// rather than of the two formulations.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>

#include "switchback/plan/refine.hpp"

#if SB_WITH_OSQP
#include <osqp.h>
#endif

namespace sb::plan {
namespace {

#if SB_WITH_OSQP

constexpr int kStateDim = 4;
constexpr int kCtrlDim = 2;
// Obstacle half-plane rows reserved per knot. Three ego discs against the
// nearest agents; beyond this the remaining pairs are handled by the cost's
// clearance term in the outer lattice rather than as hard constraints.
constexpr int kObsRowsPerStep = 3;

struct Triplet {
  int r;
  int c;
  Scalar v;
};

// Column-major CSC with a stable ordering, so the pattern can be built once and
// the values refilled in the same sequence on every subsequent cycle.
struct Csc {
  std::vector<c_int> p, i;
  std::vector<c_float> x;
  int rows{0}, cols{0};
  std::map<std::pair<int, int>, std::size_t> slot;  // (r,c) -> index into x

  void buildPattern(const std::vector<Triplet>& trips, int n_rows, int n_cols) {
    rows = n_rows;
    cols = n_cols;
    std::vector<Triplet> t = trips;
    std::sort(t.begin(), t.end(), [](const Triplet& a, const Triplet& b) {
      return a.c != b.c ? a.c < b.c : a.r < b.r;
    });
    p.assign(static_cast<std::size_t>(n_cols) + 1, 0);
    i.clear();
    x.clear();
    slot.clear();
    int col = 0;
    for (const Triplet& e : t) {
      if (!i.empty() && i.back() == e.r && col == e.c) continue;  // dedupe
      while (col < e.c) {
        p[static_cast<std::size_t>(col) + 1] = static_cast<c_int>(i.size());
        ++col;
      }
      slot[{e.r, e.c}] = i.size();
      i.push_back(e.r);
      x.push_back(0.0);
    }
    while (col < n_cols) {
      p[static_cast<std::size_t>(col) + 1] = static_cast<c_int>(i.size());
      ++col;
    }
  }

  void zero() { std::fill(x.begin(), x.end(), 0.0); }

  void set(int r, int c, Scalar v) {
    const auto it = slot.find({r, c});
    if (it != slot.end()) x[it->second] = static_cast<c_float>(v);
  }
  void add(int r, int c, Scalar v) {
    const auto it = slot.find({r, c});
    if (it != slot.end()) x[it->second] += static_cast<c_float>(v);
  }
};

#endif  // SB_WITH_OSQP

}  // namespace

#if !SB_WITH_OSQP

void QpRefiner::reserve(std::size_t) {}
bool QpRefiner::available() { return false; }
void QpRefiner::solve(const RefineProblem&, RefineResult& out, const QpOptions&) {
  out.status = "osqp_not_built";
  out.converged = false;
  out.iterations = 0;
}

#else

bool QpRefiner::available() { return true; }

namespace {

// Everything OSQP needs, owned for the lifetime of the refiner so that setup
// happens once. Kept in a file-local struct rather than in the header so that
// osqp.h stays out of the public interface.
struct QpState {
  OSQPWorkspace* work{nullptr};
  OSQPSettings* settings{nullptr};
  OSQPData* data{nullptr};
  Csc P, A;
  std::vector<c_float> q, l, u;
  int n{0}, m{0}, N{0};
  // Row offsets of each constraint block.
  int row_dyn{0}, row_x0{0}, row_ctrl{0}, row_corr{0}, row_speed{0}, row_obs{0};

  ~QpState() {
    if (work != nullptr) osqp_cleanup(work);
    if (data != nullptr) c_free(data);
    if (settings != nullptr) c_free(settings);
  }
};

std::map<const QpRefiner*, std::unique_ptr<QpState>>& qpRegistry() {
  static std::map<const QpRefiner*, std::unique_ptr<QpState>> reg;
  return reg;
}

int xIdx(int k) { return k * kStateDim; }
int uIdx(int N, int k) { return (N + 1) * kStateDim + k * kCtrlDim; }

}  // namespace

void QpRefiner::reserve(std::size_t horizon_steps) {
  const int N = static_cast<int>(horizon_steps);
  auto st = std::make_unique<QpState>();
  st->N = N;
  st->n = (N + 1) * kStateDim + N * kCtrlDim;

  // Constraint layout.
  st->row_dyn = 0;                                  // N * 4 equality
  st->row_x0 = st->row_dyn + N * kStateDim;         // 4 equality
  st->row_ctrl = st->row_x0 + kStateDim;            // N * 2 box
  st->row_corr = st->row_ctrl + N * kCtrlDim;       // (N+1) corridor
  st->row_speed = st->row_corr + (N + 1);           // (N+1) speed box
  st->row_obs = st->row_speed + (N + 1);            // (N+1) * kObsRowsPerStep
  st->m = st->row_obs + (N + 1) * kObsRowsPerStep;

  // P: block diagonal, upper triangular only, dense 4x4 state blocks and dense
  // 2x2 control blocks.
  std::vector<Triplet> pt;
  for (int k = 0; k <= N; ++k) {
    const int b = xIdx(k);
    for (int r = 0; r < kStateDim; ++r) {
      for (int c = r; c < kStateDim; ++c) pt.push_back({b + r, b + c, 0.0});
    }
  }
  for (int k = 0; k < N; ++k) {
    const int b = uIdx(N, k);
    for (int r = 0; r < kCtrlDim; ++r) {
      for (int c = r; c < kCtrlDim; ++c) pt.push_back({b + r, b + c, 0.0});
    }
  }
  st->P.buildPattern(pt, st->n, st->n);

  // A: dynamics, initial state, control box, corridor, speed box, obstacles.
  std::vector<Triplet> at;
  for (int k = 0; k < N; ++k) {
    const int row = st->row_dyn + k * kStateDim;
    for (int r = 0; r < kStateDim; ++r) {
      at.push_back({row + r, xIdx(k + 1) + r, 1.0});  // +I on dx_{k+1}
      for (int c = 0; c < kStateDim; ++c) at.push_back({row + r, xIdx(k) + c, 0.0});
      for (int c = 0; c < kCtrlDim; ++c) at.push_back({row + r, uIdx(N, k) + c, 0.0});
    }
  }
  for (int r = 0; r < kStateDim; ++r) at.push_back({st->row_x0 + r, xIdx(0) + r, 1.0});
  for (int k = 0; k < N; ++k) {
    for (int r = 0; r < kCtrlDim; ++r) {
      at.push_back({st->row_ctrl + k * kCtrlDim + r, uIdx(N, k) + r, 1.0});
    }
  }
  for (int k = 0; k <= N; ++k) {
    at.push_back({st->row_corr + k, xIdx(k) + 0, 0.0});
    at.push_back({st->row_corr + k, xIdx(k) + 1, 0.0});
    at.push_back({st->row_speed + k, xIdx(k) + 3, 1.0});
    for (int j = 0; j < kObsRowsPerStep; ++j) {
      const int row = st->row_obs + k * kObsRowsPerStep + j;
      at.push_back({row, xIdx(k) + 0, 0.0});
      at.push_back({row, xIdx(k) + 1, 0.0});
      at.push_back({row, xIdx(k) + 2, 0.0});
    }
  }
  st->A.buildPattern(at, st->m, st->n);

  st->q.assign(static_cast<std::size_t>(st->n), 0.0);
  st->l.assign(static_cast<std::size_t>(st->m), 0.0);
  st->u.assign(static_cast<std::size_t>(st->m), 0.0);
  st->P.zero();
  st->A.zero();
  // A structurally singular P would make setup fail, so seed the diagonal.
  for (int d = 0; d < st->n; ++d) st->P.set(d, d, 1.0);
  for (int r = 0; r < st->m; ++r) {
    st->l[static_cast<std::size_t>(r)] = -OSQP_INFTY;
    st->u[static_cast<std::size_t>(r)] = OSQP_INFTY;
  }

  st->settings = static_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  osqp_set_default_settings(st->settings);
  st->settings->verbose = 0;
  st->settings->warm_start = 1;
  st->settings->eps_abs = 1e-4;
  st->settings->eps_rel = 1e-4;
  st->settings->max_iter = 4000;
  st->settings->polish = 0;  // polishing roughly doubled solve time for a cost
                             // change below the tolerance, measured

  st->data = static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));
  st->data->n = st->n;
  st->data->m = st->m;
  st->data->P = csc_matrix(st->n, st->n, static_cast<c_int>(st->P.x.size()), st->P.x.data(),
                           st->P.i.data(), st->P.p.data());
  st->data->A = csc_matrix(st->m, st->n, static_cast<c_int>(st->A.x.size()), st->A.x.data(),
                           st->A.i.data(), st->A.p.data());
  st->data->q = st->q.data();
  st->data->l = st->l.data();
  st->data->u = st->u.data();

  OSQPWorkspace* w = nullptr;
  const c_int err = osqp_setup(&w, st->data, st->settings);
  st->work = (err == 0) ? w : nullptr;
  qpRegistry()[this] = std::move(st);
}

void QpRefiner::solve(const RefineProblem& prob, RefineResult& out, const QpOptions& opts) {
  out.converged = false;
  out.status = "";
  out.iterations = 0;
  out.cost = 0.0;
  out.initial_cost = 0.0;
  if (prob.reference == nullptr || prob.path == nullptr || prob.reference->pts.size() < 2) {
    out.status = "no_reference";
    return;
  }
  const auto t_start = std::chrono::steady_clock::now();
  const std::size_t n_knots = prob.reference->pts.size();
  const int N = static_cast<int>(n_knots) - 1;

  auto it = qpRegistry().find(this);
  if (it == qpRegistry().end() || it->second->N != N) {
    reserve(static_cast<std::size_t>(N));
    it = qpRegistry().find(this);
  }
  QpState& st = *it->second;
  if (st.work == nullptr) {
    out.status = "osqp_setup_failed";
    return;
  }

  const BicycleModel model{prob.veh.wheelbase};
  const auto& ref = prob.reference->pts;

  // Nominal iterate: warm start from the lattice candidate's kinematics, the
  // same starting point iLQR uses, so neither backend gets a better guess.
  if (xs_.size() != n_knots) {
    xs_.assign(n_knots, Vec4::Zero());
    x_try_.assign(n_knots, Vec4::Zero());
    x_saved_.assign(n_knots, Vec4::Zero());
    us_.assign(static_cast<std::size_t>(N), Vec2e::Zero());
    u_try_.assign(static_cast<std::size_t>(N), Vec2e::Zero());
    nrm_.assign(n_knots, Vec2{});
    refpos_.assign(n_knots, Vec2{});
  }
  for (int k = 0; k < N; ++k) {
    const auto ki = static_cast<std::size_t>(k);
    us_[ki] << clampT(ref[ki].a, prob.veh.max_decel, prob.veh.max_accel),
        clampT(std::atan(ref[ki].kappa * prob.veh.wheelbase), -prob.veh.max_steer,
               prob.veh.max_steer);
  }
  xs_[0] << prob.x0.x, prob.x0.y, prob.x0.theta, prob.x0.v;
  for (int k = 0; k < N; ++k) {
    const auto ki = static_cast<std::size_t>(k);
    xs_[ki + 1] = model.step(xs_[ki], us_[ki], prob.dt);
    xs_[ki + 1](3) = clampT(xs_[ki + 1](3), Scalar{0.0}, prob.veh.max_speed);
  }

  auto toTraj = [&](Trajectory& t) {
    t.pts.resize(n_knots);
    for (std::size_t k = 0; k < n_knots; ++k) {
      t.pts[k] = ref[k];
      t.pts[k].t = static_cast<Scalar>(k) * prob.dt;
      t.pts[k].x = xs_[k](0);
      t.pts[k].y = xs_[k](1);
      t.pts[k].heading = xs_[k](2);
      t.pts[k].v = xs_[k](3);
    }
    t.differentiate();
  };
  toTraj(cur_);
  out.initial_cost = refineObjective(prob, cur_);
  Scalar best = out.initial_cost;

  // Corridor frames, fixed across SQP passes because they are anchored to the
  // lattice candidate rather than to the iterate.
  for (std::size_t k = 0; k < n_knots; ++k) {
    const Pose2 base = prob.path->poseAt(ref[k].s);
    nrm_[k] = Vec2{std::cos(base.theta), std::sin(base.theta)}.perp();
    refpos_[k] = Vec2{ref[k].x, ref[k].y};
  }

  int iter = 0;
  for (; iter < opts.sqp_iterations; ++iter) {
    st.P.zero();
    st.A.zero();
    std::fill(st.q.begin(), st.q.end(), 0.0);
    for (int r = 0; r < st.m; ++r) {
      st.l[static_cast<std::size_t>(r)] = -OSQP_INFTY;
      st.u[static_cast<std::size_t>(r)] = OSQP_INFTY;
    }

    // Objective blocks, Gauss-Newton on the same residuals iLQR uses.
    for (int k = 0; k <= N; ++k) {
      const auto ki = static_cast<std::size_t>(k);
      const bool terminal = (k == N);
      const Scalar tw = terminal ? prob.w_terminal : 1.0;
      const int b = xIdx(k);
      const Scalar wp = 2.0 * tw * prob.w_pos;
      st.P.add(b + 0, b + 0, wp);
      st.P.add(b + 1, b + 1, wp);
      st.q[static_cast<std::size_t>(b + 0)] += static_cast<c_float>(wp * (xs_[ki](0) - ref[ki].x));
      st.q[static_cast<std::size_t>(b + 1)] += static_cast<c_float>(wp * (xs_[ki](1) - ref[ki].y));
      const Scalar wh = 2.0 * tw * prob.w_heading;
      st.P.add(b + 2, b + 2, wh);
      st.q[static_cast<std::size_t>(b + 2)] +=
          static_cast<c_float>(wh * angleDiff(xs_[ki](2), ref[ki].heading));
      const Scalar wv = 2.0 * tw * prob.w_speed;
      st.P.add(b + 3, b + 3, wv);
      st.q[static_cast<std::size_t>(b + 3)] += static_cast<c_float>(wv * (xs_[ki](3) - ref[ki].v));

      // Corridor as a two-sided linear inequality on the deviation.
      const Scalar e = nrm_[ki].dot(Vec2{xs_[ki](0), xs_[ki](1)} - refpos_[ki]);
      st.A.set(st.row_corr + k, b + 0, nrm_[ki].x);
      st.A.set(st.row_corr + k, b + 1, nrm_[ki].y);
      st.l[static_cast<std::size_t>(st.row_corr + k)] =
          static_cast<c_float>(-prob.corridor_half_width - e);
      st.u[static_cast<std::size_t>(st.row_corr + k)] =
          static_cast<c_float>(prob.corridor_half_width - e);

      st.A.set(st.row_speed + k, b + 3, 1.0);
      st.l[static_cast<std::size_t>(st.row_speed + k)] = static_cast<c_float>(-xs_[ki](3));
      st.u[static_cast<std::size_t>(st.row_speed + k)] =
          static_cast<c_float>(prob.veh.max_speed - xs_[ki](3));

      // Obstacle half-planes. Each active (disc, agent) pair contributes
      //   (diff/|diff|) . dp_disc >= R - |diff|
      // which is the first-order condition for the disc to clear the agent.
      if (prob.preds != nullptr) {
        const EgoState es{xs_[ki](0), xs_[ki](1), xs_[ki](2), xs_[ki](3)};
        Vec2 discs[3];
        Scalar rad = 0.0;
        egoDiscs(es, prob.veh, discs, rad);
        const Scalar cth = std::cos(xs_[ki](2));
        const Scalar sth = std::sin(xs_[ki](2));
        const Scalar spacing = prob.veh.length / 3.0;
        const std::size_t pk = std::min(ki, prob.preds->horizon());
        int used = 0;
        for (int di = 0; di < 3 && used < kObsRowsPerStep; ++di) {
          // Nearest agent to this disc, which is the binding constraint.
          Scalar bestd = 1e18;
          Vec2 bestdiff{};
          Scalar bestR = 0.0;
          for (std::size_t ai = 0; ai < prob.preds->size(); ++ai) {
            const auto& a = (*prob.preds)[ai];
            if (!a.validAt(pk)) continue;
            const Scalar ar = 0.5 * std::hypot(a.length, a.width);
            const Scalar R = prob.obstacle_margin + rad + ar;
            const Vec2 diff = discs[static_cast<std::size_t>(di)] - a.pos[pk];
            const Scalar d = diff.norm();
            // Only near-active pairs become constraints. Adding every pair
            // would make the QP large and mostly slack.
            if (d < R + 1.5 && d < bestd && d > 1e-6) {
              bestd = d;
              bestdiff = diff;
              bestR = R;
            }
          }
          if (bestd > 1e17) continue;
          const Vec2 g = bestdiff / bestd;
          const Scalar off = prob.veh.rear_axle_to_center + static_cast<Scalar>(di - 1) * spacing;
          const Vec2 dcdtheta{-sth * off, cth * off};
          const int row = st.row_obs + k * kObsRowsPerStep + used;
          st.A.set(row, b + 0, g.x);
          st.A.set(row, b + 1, g.y);
          st.A.set(row, b + 2, g.dot(dcdtheta));
          st.l[static_cast<std::size_t>(row)] = static_cast<c_float>(bestR - bestd);
          st.u[static_cast<std::size_t>(row)] = OSQP_INFTY;
          ++used;
        }
      }
    }

    for (int k = 0; k < N; ++k) {
      const auto ki = static_cast<std::size_t>(k);
      const int b = uIdx(N, k);
      st.P.add(b + 0, b + 0, 2.0 * prob.w_accel);
      st.P.add(b + 1, b + 1, 2.0 * prob.w_steer);
      st.q[static_cast<std::size_t>(b + 0)] +=
          static_cast<c_float>(2.0 * prob.w_accel * us_[ki](0));
      st.q[static_cast<std::size_t>(b + 1)] +=
          static_cast<c_float>(2.0 * prob.w_steer * us_[ki](1));

      // Dynamics: dx_{k+1} - A dx_k - B du_k = -(f(x_k,u_k) - x_{k+1}), which is
      // zero at the nominal because the nominal was produced by rolling out.
      Mat4 A;
      Mat42 B;
      model.jacobians(xs_[ki], us_[ki], prob.dt, A, B);
      const Vec4 defect = model.step(xs_[ki], us_[ki], prob.dt) - xs_[ki + 1];
      const int row = st.row_dyn + k * kStateDim;
      for (int r = 0; r < kStateDim; ++r) {
        st.A.set(row + r, xIdx(k + 1) + r, 1.0);
        for (int c = 0; c < kStateDim; ++c) st.A.set(row + r, xIdx(k) + c, -A(r, c));
        for (int c = 0; c < kCtrlDim; ++c) st.A.set(row + r, uIdx(N, k) + c, -B(r, c));
        st.l[static_cast<std::size_t>(row + r)] = static_cast<c_float>(-defect(r));
        st.u[static_cast<std::size_t>(row + r)] = static_cast<c_float>(-defect(r));
      }

      for (int r = 0; r < kCtrlDim; ++r) {
        st.A.set(st.row_ctrl + k * kCtrlDim + r, uIdx(N, k) + r, 1.0);
      }
      const std::size_t cr = static_cast<std::size_t>(st.row_ctrl + k * kCtrlDim);
      st.l[cr + 0] = static_cast<c_float>(prob.veh.max_decel - us_[ki](0));
      st.u[cr + 0] = static_cast<c_float>(prob.veh.max_accel - us_[ki](0));
      st.l[cr + 1] = static_cast<c_float>(-prob.veh.max_steer - us_[ki](1));
      st.u[cr + 1] = static_cast<c_float>(prob.veh.max_steer - us_[ki](1));
    }
    for (int r = 0; r < kStateDim; ++r) {
      st.A.set(st.row_x0 + r, xIdx(0) + r, 1.0);
      st.l[static_cast<std::size_t>(st.row_x0 + r)] = 0.0;
      st.u[static_cast<std::size_t>(st.row_x0 + r)] = 0.0;
    }

    osqp_update_P(st.work, st.P.x.data(), OSQP_NULL, static_cast<c_int>(st.P.x.size()));
    osqp_update_A(st.work, st.A.x.data(), OSQP_NULL, static_cast<c_int>(st.A.x.size()));
    osqp_update_lin_cost(st.work, st.q.data());
    osqp_update_bounds(st.work, st.l.data(), st.u.data());
    const c_int rc = osqp_solve(st.work);
    if (rc != 0 || st.work->info->status_val != OSQP_SOLVED) {
      // Static strings only; OSQP's own status text is copied into a fixed
      // set so that a diagnostic cannot allocate inside a planning cycle.
      switch (st.work->info->status_val) {
        case OSQP_PRIMAL_INFEASIBLE: out.status = "osqp_primal_infeasible"; break;
        case OSQP_DUAL_INFEASIBLE: out.status = "osqp_dual_infeasible"; break;
        case OSQP_MAX_ITER_REACHED: out.status = "osqp_max_iter"; break;
        default: out.status = "osqp_failed"; break;
      }
      break;
    }

    // Accept the step only if it improves the shared objective, evaluated on a
    // true nonlinear rollout. A QP step that improves the linearised model but
    // not the real problem is a step this loop must refuse, and refusing it is
    // what makes the SQP iteration count meaningful.
    for (int k = 0; k < N; ++k) {
      const auto ki = static_cast<std::size_t>(k);
      const int b = uIdx(N, k);
      u_try_[ki](0) = clampT(us_[ki](0) + st.work->solution->x[b + 0], prob.veh.max_decel,
                             prob.veh.max_accel);
      u_try_[ki](1) = clampT(us_[ki](1) + st.work->solution->x[b + 1], -prob.veh.max_steer,
                             prob.veh.max_steer);
    }
    x_try_[0] = xs_[0];
    for (int k = 0; k < N; ++k) {
      const auto ki = static_cast<std::size_t>(k);
      x_try_[ki + 1] = model.step(x_try_[ki], u_try_[ki], prob.dt);
      x_try_[ki + 1](3) = clampT(x_try_[ki + 1](3), Scalar{0.0}, prob.veh.max_speed);
    }
    x_saved_ = xs_;
    xs_ = x_try_;
    toTraj(cand_);
    const Scalar c = refineObjective(prob, cand_);
    if (c < best) {
      best = c;
      us_ = u_try_;
      cur_ = cand_;
      out.converged = true;
    } else {
      xs_ = x_saved_;
      break;
    }
  }

  if (out.status[0] == '\0') out.status = out.converged ? "converged" : "no_improvement";
  out.iterations = iter + 1;
  out.cost = best;
  out.traj.pts = cur_.pts;
  for (TrajPoint& p : out.traj.pts) {
    const FrenetPoint f = prob.path->toFrenet(Vec2{p.x, p.y}, p.heading, p.v);
    p.s = f.s;
    p.d = f.d;
  }

  out.max_corridor_violation = 0.0;
  out.min_obstacle_margin = 999.0;
  for (std::size_t k = 0; k < out.traj.pts.size(); ++k) {
    const Scalar e = nrm_[k].dot(Vec2{out.traj.pts[k].x, out.traj.pts[k].y} - refpos_[k]);
    out.max_corridor_violation =
        std::max(out.max_corridor_violation, std::abs(e) - prob.corridor_half_width);
    if (prob.preds != nullptr) {
      const Box2 ego =
          egoFootprint(out.traj.pts[k].x, out.traj.pts[k].y, out.traj.pts[k].heading, prob.veh);
      const std::size_t pk = std::min(k, prob.preds->horizon());
      for (std::size_t ai = 0; ai < prob.preds->size(); ++ai) {
        const auto& a = (*prob.preds)[ai];
        if (!a.validAt(pk)) continue;
        out.min_obstacle_margin = std::min(out.min_obstacle_margin, obbDistance(ego, a.boxAt(pk)));
      }
    }
  }
  out.max_corridor_violation = std::max(Scalar{0.0}, out.max_corridor_violation);
  out.solve_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
}

#endif  // SB_WITH_OSQP

}  // namespace sb::plan
