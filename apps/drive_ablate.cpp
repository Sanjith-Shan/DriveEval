// Root-cause attribution for a failing scenario.
//
// The req's first bullet is "investigate driving behaviour issues from logs and
// understand the root cause". A list of scenarios that failed does not do that.
// This tool re-runs a failing scenario with one thing changed at a time and
// reports which changes fix it, which turns "the ego collided" into "the ego
// collided because it mispredicted agent 14, and perfect prediction fixes it".
//
// Every ablation is a counterfactual on the same scenario, so the comparison is
// paired by construction and needs no statistics.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"
#include "driveeval/sim/simulator.hpp"

using namespace drive;

namespace {

struct Outcome {
  bool collision{false};
  bool at_fault{false};
  bool offroad{false};
  Scalar min_ttc{0.0};
  Scalar progress{0.0};

  static Outcome of(const eval::ScenarioMetrics& m) {
    Outcome o;
    o.collision = m.collision != 0;
    o.at_fault = m.at_fault_collision != 0;
    o.offroad = m.drivable_area_violation > 0;
    o.min_ttc = m.min_ttc;
    o.progress = m.progress_ratio;
    return o;
  }
  [[nodiscard]] bool failed() const { return collision || offroad; }
};

const char* verdict(const Outcome& base, const Outcome& ab) {
  if (base.failed() && !ab.failed()) return "FIXES";
  if (!base.failed() && ab.failed()) return "CAUSES";
  return "-";
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  args.requireNoUnknown({"shard", "scenario", "id", "mode", "backend", "csv", "max-agents",
                         "quiet"});
  const std::string path = args.str("shard");
  if (path.empty()) {
    std::fprintf(stderr,
                 "usage: drive_ablate --shard F [--scenario N | --id ID] [--mode reactive]\n"
                 "                 [--backend ilqr] [--csv out.csv]\n");
    return 2;
  }
  auto shard = io::ShardReader::open(path);
  if (!shard) {
    std::fprintf(stderr, "error: %s\n", shard.error().c_str());
    return 1;
  }
  std::size_t index = static_cast<std::size_t>(args.integer("scenario", 0));
  const std::string want = args.str("id");
  if (!want.empty()) {
    bool found = false;
    for (std::size_t i = 0; i < shard->size(); ++i) {
      if (shard->scenarioId(i) == want) {
        index = i;
        found = true;
        break;
      }
    }
    if (!found) {
      std::fprintf(stderr, "scenario %s not found\n", want.c_str());
      return 1;
    }
  }
  auto sv = shard->scenario(index);
  if (!sv) {
    std::fprintf(stderr, "error: %s\n", sv.error().c_str());
    return 1;
  }

  sim::SimConfig base_cfg;
  base_cfg.agent_mode = sim::agentModeFromString(args.str("mode", "reactive"));
  base_cfg.planner.backend = plan::backendFromString(args.str("backend", "ilqr"));
  base_cfg.planner.cost_ctx.capabilities = shard->capabilities();

  auto run = [&](const sim::SimConfig& c) {
    sim::Simulator s;
    s.setup(c);
    const sim::SimOutput& o = s.run(*sv, shard->capabilities());
    return std::pair<Outcome, const char*>{Outcome::of(o.metrics), o.metrics.status};
  };

  const auto [base, base_status] = run(base_cfg);
  std::printf("scenario        %s\n", sv->id().c_str());
  std::printf("baseline        %s  collision=%d at_fault=%d offroad=%d min_ttc=%.2f progress=%.3f\n",
              base_status, base.collision, base.at_fault, base.offroad, base.min_ttc,
              base.progress);
  if (std::string(base_status) != "ok") {
    std::printf("\nnot ok, nothing to attribute\n");
    return 0;
  }

  struct Row {
    std::string kind;
    std::string detail;
    Outcome out;
  };
  std::vector<Row> rows;

  // 1. Which agent caused it. Only agents that came near the ego are worth
  //    removing, so the sweep is over the ones that could plausibly matter.
  const auto ego_track = sv->egoTrack();
  std::vector<std::pair<Scalar, std::size_t>> near;
  for (std::size_t a = 0; a < sv->numAgents(); ++a) {
    if (a == static_cast<std::size_t>(sv->egoIndex())) continue;
    Scalar best = 1e18;
    for (std::size_t t = 0; t < sv->numSteps(); ++t) {
      const io::AgentState& s = sv->state(a, t);
      if (s.valid == 0 || ego_track[t].valid == 0) continue;
      best = std::min(best, std::hypot(static_cast<Scalar>(s.x) - ego_track[t].x,
                                       static_cast<Scalar>(s.y) - ego_track[t].y));
    }
    if (best < 25.0) near.emplace_back(best, a);
  }
  std::sort(near.begin(), near.end());
  const auto max_agents = static_cast<std::size_t>(args.integer("max-agents", 8));
  if (near.size() > max_agents) near.resize(max_agents);
  for (const auto& [d, a] : near) {
    sim::SimConfig c = base_cfg;
    c.suppress_agent = static_cast<int>(a);
    rows.push_back({"remove_agent", "agent " + std::to_string(a) + " (closest " +
                                        std::to_string(static_cast<int>(d)) + " m)",
                    run(c).first});
  }

  // 2. Perfect prediction. Separates a planning failure from a prediction
  //    failure, which are different bugs with different owners.
  {
    sim::SimConfig c = base_cfg;
    c.planner.prediction = predict::Model::kLoggedOracle;
    rows.push_back({"prediction", "logged oracle", run(c).first});
  }
  {
    sim::SimConfig c = base_cfg;
    c.planner.prediction = predict::Model::kLaneFollow;
    rows.push_back({"prediction", "lane follow", run(c).first});
  }

  // 3. Refinement. Did the optimiser make it worse than the lattice candidate?
  {
    sim::SimConfig c = base_cfg;
    c.planner.backend = plan::Backend::kNone;
    rows.push_back({"refinement", "disabled", run(c).first});
  }

  // 4. Agent model. Whether the failure is an artefact of how other agents were
  //    simulated rather than of the plan.
  {
    sim::SimConfig c = base_cfg;
    c.agent_mode = base_cfg.agent_mode == sim::AgentMode::kReactive ? sim::AgentMode::kLogReplay
                                                                   : sim::AgentMode::kReactive;
    rows.push_back({"agent_model", sim::toString(c.agent_mode), run(c).first});
  }

  // 5. Cost terms, one at a time. A term whose removal fixes the failure was
  //    steering the planner into it; a term whose removal causes one was
  //    preventing it.
  for (std::size_t t = 0; t < plan::kNumCostTerms; ++t) {
    const auto term = static_cast<plan::CostTerm>(t);
    if (base_cfg.planner.weights[term] == 0.0) continue;
    sim::SimConfig c = base_cfg;
    c.planner.weights[term] = 0.0;
    rows.push_back({"zero_cost_term", plan::toString(term), run(c).first});
  }

  std::printf("\n%-16s %-34s %-9s %-9s %-8s %-8s %s\n", "ablation", "detail", "collision",
              "at_fault", "offroad", "progress", "verdict");
  for (const Row& r : rows) {
    std::printf("%-16s %-34s %-9d %-9d %-8d %-8.3f %s\n", r.kind.c_str(), r.detail.c_str(),
                r.out.collision, r.out.at_fault, r.out.offroad, r.out.progress,
                verdict(base, r.out));
  }

  const std::string csv = args.str("csv");
  if (!csv.empty()) {
    std::FILE* f = std::fopen(csv.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "scenario_id,kind,detail,collision,at_fault,offroad,min_ttc,progress,"
                      "base_collision,base_at_fault,base_offroad,verdict\n");
      for (const Row& r : rows) {
        std::fprintf(f, "%s,%s,\"%s\",%d,%d,%d,%.4f,%.4f,%d,%d,%d,%s\n", sv->id().c_str(),
                     r.kind.c_str(), r.detail.c_str(), r.out.collision, r.out.at_fault,
                     r.out.offroad, r.out.min_ttc, r.out.progress, base.collision, base.at_fault,
                     base.offroad, verdict(base, r.out));
      }
      std::fclose(f);
      std::printf("\ncsv             %s\n", csv.c_str());
    }
  }
  return 0;
}
