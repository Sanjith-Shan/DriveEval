// Criticality sweeps: how close was this scenario to going the other way.
//
// The req's fourth bullet is the long tail. A scenario that passed is not the
// same as a scenario that was never going to fail, and a scenario that failed
// by a hair is not the same as one that failed by a mile. This tool perturbs a
// single interaction along one axis -- when the ego meets a given agent, or how
// fast that agent was going -- and finds the value at which the outcome flips.
//
// The distance from the recorded value to that boundary is the scenario's
// margin. A fleet with a small median margin is one bad day from a different
// safety record, and no aggregate collision rate says that.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "driveeval/sim/simulator.hpp"

using namespace drive;

namespace {

struct SweepPoint {
  Scalar value{0.0};
  bool failed{false};
  Scalar min_ttc{0.0};
  Scalar progress{0.0};
};

bool failedOf(const eval::ScenarioMetrics& m) {
  return m.collision != 0 || m.drivable_area_violation > 0;
}

// The agent worth perturbing: the one that came closest to the ego.
int closestAgent(const io::ScenarioView& sv, Scalar& out_dist) {
  const auto ego = sv.egoTrack();
  int best = -1;
  Scalar best_d = 1e18;
  for (std::size_t a = 0; a < sv.numAgents(); ++a) {
    if (a == static_cast<std::size_t>(sv.egoIndex())) continue;
    const io::u32 type = sv.agents[a].type;
    if (type == io::kAgentStatic || type == io::kAgentBackground ||
        type == io::kAgentConstruction) {
      continue;
    }
    for (std::size_t t = 0; t < sv.numSteps(); ++t) {
      const io::AgentState& s = sv.state(a, t);
      if (s.valid == 0 || ego[t].valid == 0) continue;
      const Scalar d = std::hypot(static_cast<Scalar>(s.x) - ego[t].x,
                                  static_cast<Scalar>(s.y) - ego[t].y);
      if (d < best_d) {
        best_d = d;
        best = static_cast<int>(a);
      }
    }
  }
  out_dist = best_d;
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  args.requireNoUnknown({"shards", "scenario", "id", "mode", "backend", "csv", "limit", "axis",
                         "range", "step", "quiet"});
  const std::string shards = args.str("shards");
  if (shards.empty()) {
    std::fprintf(stderr,
                 "usage: drive_sweep --shards F[,F...] [--id ID | --limit N] [--axis time|speed]\n"
                 "                [--range 20] [--step 2] [--mode reactive] [--csv out.csv]\n");
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

  sim::SimConfig base;
  base.agent_mode = sim::agentModeFromString(args.str("mode", "reactive"));
  base.planner.backend = plan::backendFromString(args.str("backend", "ilqr"));
  const std::string axis = args.str("axis", "time");
  const auto range = static_cast<int>(args.integer("range", 20));
  const auto step = std::max(1, static_cast<int>(args.integer("step", 2)));
  const auto limit = static_cast<std::size_t>(args.integer("limit", 25));
  const std::string want = args.str("id");
  const bool quiet = args.has("quiet");

  std::FILE* f = nullptr;
  const std::string csv = args.str("csv");
  if (!csv.empty()) {
    f = std::fopen(csv.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "scenario_id,axis,agent,closest_m,value,failed,min_ttc,progress\n");
    }
  }

  sim::Simulator simulator;
  std::size_t done = 0, flipped = 0, always_ok = 0, always_bad = 0;
  std::vector<double> margins;

  for (const std::string& p : paths) {
    auto shard = io::ShardReader::open(p);
    if (!shard) continue;
    base.planner.cost_ctx.capabilities = shard->capabilities();
    for (std::size_t i = 0; i < shard->size() && done < limit; ++i) {
      if (!want.empty() && shard->scenarioId(i) != want) continue;
      auto sv = shard->scenario(i);
      if (!sv || !sv->hasEgo()) continue;
      Scalar closest = 0.0;
      const int agent = closestAgent(*sv, closest);
      if (agent < 0) continue;

      std::vector<SweepPoint> pts;
      bool nominal_failed = false;
      for (int k = -range; k <= range; k += step) {
        sim::SimConfig c = base;
        c.perturb.agent = agent;
        if (axis == "speed") {
          c.perturb.speed_scale = 1.0 + 0.05 * static_cast<Scalar>(k);
        } else {
          c.perturb.time_shift = k;
        }
        simulator.setup(c);
        const sim::SimOutput& o = simulator.run(*sv, shard->capabilities());
        if (!o.ok) continue;
        SweepPoint sp;
        sp.value = axis == "speed" ? 1.0 + 0.05 * static_cast<Scalar>(k)
                                   : static_cast<Scalar>(k) * sv->dt();
        sp.failed = failedOf(o.metrics);
        sp.min_ttc = o.metrics.min_ttc;
        sp.progress = o.metrics.progress_ratio;
        if (k == 0) nominal_failed = sp.failed;
        pts.push_back(sp);
        if (f != nullptr) {
          std::fprintf(f, "%s,%s,%d,%.2f,%.4f,%d,%.4f,%.4f\n", sv->id().c_str(), axis.c_str(),
                       agent, closest, sp.value, sp.failed ? 1 : 0, sp.min_ttc, sp.progress);
        }
      }
      if (pts.empty()) continue;
      ++done;

      // The margin: how far the perturbation has to move from the recorded
      // value before the outcome changes. Unbounded when it never does.
      Scalar margin = 1e18;
      for (const SweepPoint& sp : pts) {
        if (sp.failed != nominal_failed) {
          margin = std::min(margin, std::abs(sp.value - (axis == "speed" ? 1.0 : 0.0)));
        }
      }
      std::size_t n_fail = 0;
      for (const SweepPoint& sp : pts) {
        if (sp.failed) ++n_fail;
      }
      if (n_fail == 0) {
        ++always_ok;
      } else if (n_fail == pts.size()) {
        ++always_bad;
      } else {
        ++flipped;
        margins.push_back(margin);
      }

      if (!quiet) {
        std::printf("%s  agent %3d (closest %5.1f m)  nominal %s  fail %2zu/%2zu  ",
                    sv->id().c_str(), agent, closest, nominal_failed ? "FAIL" : "ok  ", n_fail,
                    pts.size());
        if (margin < 1e17) {
          std::printf("margin %.2f %s\n", margin, axis == "speed" ? "x" : "s");
        } else {
          std::printf("margin none in range\n");
        }
      }
    }
  }
  if (f != nullptr) std::fclose(f);

  std::sort(margins.begin(), margins.end());
  std::printf("\nswept          %zu scenarios on the %s axis, +-%d steps\n", done, axis.c_str(),
              range);
  std::printf("  never fails  %zu\n", always_ok);
  std::printf("  always fails %zu\n", always_bad);
  std::printf("  flips        %zu  <- the scenarios whose outcome is decided by the perturbation\n",
              flipped);
  if (!margins.empty()) {
    std::printf("  margin to the boundary: p50 %.2f  p90 %.2f  min %.2f %s\n",
                margins[margins.size() / 2], margins[(margins.size() * 9) / 10], margins.front(),
                axis == "speed" ? "x" : "s");
  }
  if (!csv.empty()) std::printf("  csv %s\n", csv.c_str());
  return 0;
}
