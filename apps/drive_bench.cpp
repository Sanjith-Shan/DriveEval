// Latency, allocation behaviour, and the iLQR-versus-OSQP head to head.
//
// The comparison is matched: the simulation is driven by one backend, and at
// every cycle the *same* RefineProblem is handed to each solver in turn. They
// therefore see identical initial states, identical lattice candidates and
// identical predictions, so a difference in the reported objective is a
// difference between the formulations rather than between the situations they
// happened to be given.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "driveeval/build_info.hpp"
#include "driveeval/core/alloc_count.hpp"
#include "driveeval/sim/simulator.hpp"

using namespace drive;

namespace {

struct Stat {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  double pct(double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(pos);
    const auto hi = std::min(v.size() - 1, lo + 1);
    const double f = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - f) + v[hi] * f;
  }
  double mean() const {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (const double x : v) s += x;
    return s / static_cast<double>(v.size());
  }
  std::size_t n() const { return v.size(); }
};

struct BackendStats {
  Stat solve_us;
  Stat iterations;
  Stat cost_reduction;   // fraction of the initial objective removed
  Stat corridor_violation;
  std::size_t converged{0};
  std::size_t worse{0};   // ended above the objective it started from
  std::size_t problems{0};
};

void report(const char* name, BackendStats& b) {
  if (b.problems == 0) {
    std::printf("  %-6s  not run\n", name);
    return;
  }
  std::printf("  %-6s  solve p50 %7.1f us  p99 %8.1f us   iters %5.2f   "
              "cost -%5.1f%%   converged %5.1f%%   corridor p99 %.3f m\n",
              name, b.solve_us.pct(0.50), b.solve_us.pct(0.99), b.iterations.mean(),
              100.0 * b.cost_reduction.mean(),
              100.0 * static_cast<double>(b.converged) / static_cast<double>(b.problems),
              b.corridor_violation.pct(0.99));
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  args.requireNoUnknown({"shards", "limit", "mode", "drive-with", "csv", "warmup"});
  const std::string shards = args.str("shards");
  if (shards.empty()) {
    std::fprintf(stderr, "usage: drive_bench --shards F[,F...] [--limit N] [--mode reactive]\n");
    return 2;
  }

  std::vector<std::string> paths;
  {
    std::string cur;
    for (const char c : shards) {
      if (c == ',') {
        if (!cur.empty()) paths.push_back(cur);
        cur.clear();
      } else {
        cur += c;
      }
    }
    if (!cur.empty()) paths.push_back(cur);
  }

  sim::SimConfig cfg;
  cfg.agent_mode = sim::agentModeFromString(args.str("mode", "reactive"));
  cfg.planner.backend = plan::backendFromString(args.str("drive-with", "ilqr"));
  cfg.count_allocs = true;

  const auto limit = static_cast<std::size_t>(args.integer("limit", 200));

  sim::Simulator simulator;
  simulator.setup(cfg);
  plan::IlqrRefiner ilqr;
  plan::QpRefiner qp;
  ilqr.reserve(cfg.planner.lattice.horizonSteps());
  if (plan::QpRefiner::available()) qp.reserve(cfg.planner.lattice.horizonSteps());
  plan::RefineResult res_i, res_q;

  BackendStats bi, bq;
  Stat cycle_us, lattice_us, cost_us;
  std::int64_t hot_allocs = 0;
  std::size_t scenarios = 0, cycles = 0;
  io::u32 caps = 0;

  const auto t0 = std::chrono::steady_clock::now();
  for (const std::string& p : paths) {
    auto shard = io::ShardReader::open(p);
    if (!shard) continue;
    caps = shard->capabilities();
    cfg.planner.cost_ctx.capabilities = caps;
    for (std::size_t i = 0; i < shard->size() && scenarios < limit; ++i) {
      auto sv = shard->scenario(i);
      if (!sv) continue;
      const sim::SimOutput& out = simulator.run(*sv, caps);
      if (!out.ok) continue;
      ++scenarios;
      for (const auto& c : out.cycles) {
        cycle_us.add(c.plan_us);
        ++cycles;
      }
      hot_allocs += out.metrics.hot_path_allocs;

      // Matched head to head. Re-drive the scenario one cycle at a time and
      // hand both solvers the identical problem at a sample of cycles.
      plan::Planner& planner = simulator.planner();
      for (std::size_t k = 0; k < out.cycles.size(); k += 7) {
        const auto& c = out.cycles[k];
        const plan::PlanOutput& po = planner.plan(c.ego, c.a_lon, c.step);
        if (!po.ok || po.chosen_candidate == nullptr) continue;
        lattice_us.add(po.lattice_us);
        cost_us.add(po.cost_us);

        plan::RefineProblem prob = cfg.planner.refine;
        prob.reference = po.chosen_candidate;
        prob.path = &simulator.path();
        prob.preds = &planner.predictions();
        prob.x0 = c.ego;
        prob.dt = cfg.planner.lattice.dt;
        prob.corridor_half_width = cfg.planner.corridor_half_width;
        prob.veh = cfg.planner.veh;

        ilqr.solve(prob, res_i, cfg.planner.ilqr);
        ++bi.problems;
        bi.solve_us.add(res_i.solve_us);
        bi.iterations.add(res_i.iterations);
        bi.cost_reduction.add(res_i.costReduction());
        bi.corridor_violation.add(res_i.max_corridor_violation);
        if (res_i.converged) ++bi.converged;
        if (res_i.cost > res_i.initial_cost + 1e-9) ++bi.worse;

        if (plan::QpRefiner::available()) {
          qp.solve(prob, res_q, cfg.planner.qp);
          ++bq.problems;
          bq.solve_us.add(res_q.solve_us);
          bq.iterations.add(res_q.iterations);
          bq.cost_reduction.add(res_q.costReduction());
          bq.corridor_violation.add(res_q.max_corridor_violation);
          if (res_q.converged) ++bq.converged;
          if (res_q.cost > res_q.initial_cost + 1e-9) ++bq.worse;
        }
      }
    }
    if (scenarios >= limit) break;
  }
  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("driveeval benchmark\n");
  std::printf("build           %s %s %s, OSQP %s\n", kBuildType, kCompiler, kArchFlag,
              kWithOsqp ? "on" : "off");
  std::printf("hardware        Apple M3 Pro (12 core), process not pinned, single thread\n");
  std::printf("mode            %s, driven with %s\n", sim::toString(cfg.agent_mode),
              plan::toString(cfg.planner.backend));
  std::printf("scenarios       %zu, cycles %zu, wall %.1f s\n", scenarios, cycles, wall);
  std::printf("\nplanning cycle (whole cycle, 10 Hz budget is 100000 us)\n");
  std::printf("  p50 %.0f us   p90 %.0f us   p99 %.0f us   max %.0f us\n", cycle_us.pct(0.50),
              cycle_us.pct(0.90), cycle_us.pct(0.99), cycle_us.pct(1.0));
  std::printf("  The median is the honest number. The tail carries OS scheduling noise on an\n"
              "  unpinned process and is reported rather than trimmed.\n");
  std::printf("\nphase breakdown (mean)\n");
  std::printf("  lattice %.0f us   cost %.0f us\n", lattice_us.mean(), cost_us.mean());
  std::printf("\nheap allocations inside the planning cycle: %lld over %zu cycles\n",
              static_cast<long long>(hot_allocs), cycles);
  std::printf("\nrefinement, matched problems (%zu each)\n", bi.problems);
  report("iLQR", bi);
  report("OSQP", bq);
  std::printf("\n  ended worse than they started: iLQR %zu, OSQP %zu (both must be 0)\n", bi.worse,
              bq.worse);

  const std::string csv = args.str("csv");
  if (!csv.empty()) {
    std::FILE* f = std::fopen(csv.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "backend,problems,solve_us_p50,solve_us_p99,iters_mean,cost_reduction_mean,"
                      "converged_frac,corridor_p99,ended_worse\n");
      auto row = [&](const char* n, BackendStats& b) {
        if (b.problems == 0) return;
        std::fprintf(f, "%s,%zu,%.2f,%.2f,%.3f,%.5f,%.4f,%.5f,%zu\n", n, b.problems,
                     b.solve_us.pct(0.50), b.solve_us.pct(0.99), b.iterations.mean(),
                     b.cost_reduction.mean(),
                     static_cast<double>(b.converged) / static_cast<double>(b.problems),
                     b.corridor_violation.pct(0.99), b.worse);
      };
      row("ilqr", bi);
      row("osqp", bq);
      std::fclose(f);
      std::printf("\ncsv             %s\n", csv.c_str());
    }
  }
  return 0;
}
