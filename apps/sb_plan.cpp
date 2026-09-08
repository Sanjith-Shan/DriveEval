// Run one scenario closed-loop and optionally dump it for the visualiser.
#include <cstdio>
#include <string>

#include "cli.hpp"
#include "json.hpp"
#include "switchback/build_info.hpp"
#include "switchback/sim/simulator.hpp"

using namespace sb;

namespace {

const char* agentTypeName(io::u32 t) {
  switch (t) {
    case io::kAgentVehicle: return "vehicle";
    case io::kAgentPedestrian: return "pedestrian";
    case io::kAgentCyclist: return "cyclist";
    case io::kAgentMotorcyclist: return "motorcyclist";
    case io::kAgentBus: return "bus";
    case io::kAgentStatic: return "static";
    case io::kAgentBackground: return "background";
    case io::kAgentConstruction: return "construction";
    default: return "unknown";
  }
}

const char* cityName(io::u32 c) {
  static const char* kNames[] = {"austin",        "miami",         "pittsburgh",
                                 "dearborn",      "washington-dc", "palo-alto"};
  return c < 6 ? kNames[c] : "unknown";
}

void dumpJson(const std::string& path, const io::ScenarioView& view, io::u32 caps,
              const sim::Simulator& simulator, const sim::SimOutput& out, long plan_stride) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return;
  }
  json::Writer w(f);
  const auto& graph = simulator.graph();
  const auto& refpath = simulator.path();
  const auto& world = simulator.world();
  const auto& cfg = simulator.config();

  w.beginObject();
  w.kv("schema", 1L);
  w.kv("scenario_id", view.id());
  w.kv("dataset", "av2_val");
  w.kv("city", cityName(view.header->city));
  w.kv("agent_mode", sim::toString(cfg.agent_mode));
  w.kv("planner", cfg.replay_logged_ego ? "logged_ego"
              : cfg.pure_pursuit_only ? "pure_pursuit" : plan::toString(cfg.planner.backend));
  w.kvRaw("config", cfg.planner.toJson());
  w.kv("dt", static_cast<double>(view.dt()));
  w.key("origin");
  w.beginArray();
  w.value(static_cast<double>(view.header->origin_x));
  w.value(static_cast<double>(view.header->origin_y));
  w.endArray();
  w.kv("capabilities", static_cast<long>(caps));
  // The ego's footprint is not an agent record, so it has to be stated here or
  // every consumer invents its own constant.
  w.kv("ego_length", cfg.planner.veh.length);
  w.kv("ego_width", cfg.planner.veh.width);
  w.kv("ego_rear_axle_to_center", cfg.planner.veh.rear_axle_to_center);
  // agents[].states[i] is timestep i, aligned with ego[] and ego_logged[], full
  // length, with gaps carried by `valid` rather than by omission.
  w.kv("states_are_dense", true);
  w.kv("num_steps", static_cast<long>(view.numSteps()));

  w.key("lanes");
  w.beginArray();
  for (const auto& lane : graph.lanes()) {
    if (lane.points.size() < 2) continue;
    w.beginObject();
    w.kv("id", static_cast<unsigned long>(lane.id));
    w.kv("is_intersection", lane.is_intersection);
    w.kv("lane_type", static_cast<long>(lane.lane_type));
    w.kv("speed_prior", lane.speed_prior);
    w.key("centerline");
    w.beginArray();
    for (const Vec2& p : lane.points) w.xy(p.x, p.y);
    w.endArray();
    w.endObject();
  }
  w.endArray();

  w.key("polygons");
  w.beginArray();
  for (std::size_t i = 0; i < view.polygons.size(); ++i) {
    w.beginObject();
    w.kv("kind", view.polygons[i].kind == io::kPolyCrosswalk ? "crosswalk" : "drivable_area");
    w.key("points");
    w.beginArray();
    for (const io::PointRec& p : view.polygonPoints(i)) w.xy(p.x, p.y);
    w.endArray();
    w.endObject();
  }
  w.endArray();

  w.key("route");
  w.beginArray();
  for (const std::size_t li : out.route.lanes) {
    w.value(static_cast<unsigned long>(graph.lane(li).id));
  }
  w.endArray();

  w.key("reference_path");
  w.beginArray();
  for (std::size_t i = 0; i < refpath.size(); ++i) {
    w.beginObject();
    w.kv("s", static_cast<double>(i) * refpath.step());
    w.kv("x", refpath.points()[i].x);
    w.kv("y", refpath.points()[i].y);
    w.kv("heading", refpath.headings()[i]);
    w.kv("kappa", refpath.curvatures()[i]);
    w.endObject();
  }
  w.endArray();

  w.key("ego");
  w.beginArray();
  for (const auto& c : out.cycles) {
    w.beginObject();
    w.kv("t", c.t);
    w.kv("x", c.ego.x);
    w.kv("y", c.ego.y);
    w.kv("heading", c.ego.theta);
    w.kv("v", c.ego.v);
    w.kv("a", c.a_lon);
    w.kv("plan_us", c.plan_us);
    w.endObject();
  }
  w.endArray();

  w.key("ego_logged");
  w.beginArray();
  {
    const std::size_t ei = world.egoIndex();
    for (std::size_t t = 0; t < view.numSteps(); ++t) {
      if (!world.loggedValid(ei, t)) continue;
      w.beginObject();
      w.kv("t", static_cast<double>(t) * static_cast<double>(view.dt()));
      const Vec2 p = world.loggedPos(ei, t);
      w.kv("x", p.x);
      w.kv("y", p.y);
      w.kv("heading", world.loggedHeading(ei, t));
      w.kv("v", world.loggedSpeed(ei, t));
      w.endObject();
    }
  }
  w.endArray();

  // Agents as their logged tracks. The reactive rollout diverges from these,
  // which is exactly what the visualiser needs to show, so both are written.
  w.key("agents");
  w.beginArray();
  for (std::size_t i = 0; i < view.numAgents(); ++i) {
    if (i == world.egoIndex()) continue;
    w.beginObject();
    w.kv("id", static_cast<unsigned long>(view.agents[i].id_hash));
    w.kv("type", agentTypeName(view.agents[i].type));
    w.kv("length", static_cast<double>(view.agents[i].length));
    w.kv("width", static_cast<double>(view.agents[i].width));
    w.key("states");
    w.beginArray();
    for (std::size_t t = 0; t < view.numSteps(); ++t) {
      const io::AgentState& s = view.state(i, t);
      w.beginObject();
      w.kv("t", static_cast<double>(t) * static_cast<double>(view.dt()));
      w.kv("x", static_cast<double>(s.x));
      w.kv("y", static_cast<double>(s.y));
      w.kv("heading", static_cast<double>(s.heading));
      w.kv("v", std::hypot(static_cast<double>(s.vx), static_cast<double>(s.vy)));
      w.kv("valid", s.valid != 0);
      w.endObject();
    }
    w.endArray();
    w.endObject();
  }
  w.endArray();

  // Candidate sets are written every `plan_stride` cycles: writing all sixty
  // candidates for all hundred-odd cycles dominates the file size and nothing
  // reads more than a sample of them.
  w.key("plans");
  w.beginArray();
  for (const sim::PlanSnapshot& snap : out.plans) {
    w.beginObject();
    w.kv("t", snap.t);
    w.kv("chosen_index", snap.chosen);
    w.kv("refine_used", snap.refine_used);
    w.key("candidates");
    w.beginArray();
    for (std::size_t ci = 0; ci < snap.candidates.size(); ++ci) {
      w.beginObject();
      w.kv("cost", snap.costs[ci]);
      w.kv("feasible", snap.feasible[ci] != 0);
      w.key("points");
      w.beginArray();
      for (const TrajPoint& tp : snap.candidates[ci].pts) w.xy(tp.x, tp.y);
      w.endArray();
      w.endObject();
    }
    w.endArray();
    if (snap.chosen >= 0 && static_cast<std::size_t>(snap.chosen) < snap.candidates.size()) {
      w.key("chosen");
      w.beginArray();
      for (const TrajPoint& tp : snap.candidates[static_cast<std::size_t>(snap.chosen)].pts) {
        w.xy(tp.x, tp.y);
      }
      w.endArray();
    }
    if (snap.refine_used && !snap.refined.pts.empty()) {
      w.key("refined");
      w.beginArray();
      for (const TrajPoint& tp : snap.refined.pts) w.xy(tp.x, tp.y);
      w.endArray();
    }
    w.endObject();
  }
  w.endArray();
  w.kv("plan_stride", plan_stride);

  w.key("events");
  w.beginArray();
  for (const auto& c : out.cycles) {
    if (c.collision_agent < 0) continue;
    w.beginObject();
    w.kv("t", c.t);
    w.kv("kind", c.ego_struck_from_behind ? "collision_not_at_fault" : "at_fault_collision");
    w.kv("agent", static_cast<long>(c.collision_agent));
    // Events carry their own location: at a collision it coincides with the ego
    // pose, but an off-road violation happens at a corner, not at the centre.
    w.kv("x", c.ego.x);
    w.kv("y", c.ego.y);
    w.kv("detail", agentTypeName(static_cast<io::u32>(c.collision_agent_type)));
    w.endObject();
    break;
  }
  w.endArray();

  const auto& m = out.metrics;
  w.key("metrics");
  w.beginObject();
  w.kv("status", m.status);
  w.kv("collision", m.collision);
  w.kv("at_fault_collision", m.at_fault_collision);
  w.kv("drivable_area_violation", m.drivable_area_violation);
  w.kv("wrong_direction", m.wrong_direction);
  w.kv("min_ttc", m.min_ttc);
  w.kv("progress_ratio", m.progress_ratio);
  w.kv("route_completion", m.route_completion);
  w.kv("max_abs_a_lon", m.max_abs_a_lon);
  w.kv("max_abs_a_lat", m.max_abs_a_lat);
  w.kv("max_abs_jerk", m.max_abs_jerk);
  w.kv("comfort_violation", m.comfort_violation);
  w.kv("ade", m.ade);
  w.kv("fde", m.fde);
  w.kv("plan_us_p50", m.plan_us_p50);
  w.kv("plan_us_p99", m.plan_us_p99);
  w.endObject();
  w.endObject();
  std::fputc('\n', f);
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  args.requireNoUnknown({"shard", "scenario", "id", "mode", "backend", "prediction", "dump",
                         "dump-plan-stride", "pure-pursuit", "logged-ego", "weight", "quiet", "max-steps", "trace"});

  const std::string shard_path = args.str("shard");
  if (shard_path.empty()) {
    std::fprintf(stderr,
                 "usage: sb_plan --shard F [--scenario N | --id ID] [--mode log_replay|reactive]\n"
                 "               [--backend ilqr|osqp|none] [--prediction constant_velocity|"
                 "lane_follow|logged_oracle]\n"
                 "               [--pure-pursuit] [--weight name=value] [--dump out.json]\n");
    return 2;
  }
  auto shard = io::ShardReader::open(shard_path);
  if (!shard) {
    std::fprintf(stderr, "error: %s\n", shard.error().c_str());
    return 1;
  }

  std::size_t index = static_cast<std::size_t>(args.integer("scenario", 0));
  const std::string want_id = args.str("id");
  if (!want_id.empty()) {
    bool found = false;
    for (std::size_t i = 0; i < shard->size(); ++i) {
      if (shard->scenarioId(i) == want_id) {
        index = i;
        found = true;
        break;
      }
    }
    if (!found) {
      std::fprintf(stderr, "scenario id %s not in %s\n", want_id.c_str(), shard_path.c_str());
      return 1;
    }
  }
  auto sv = shard->scenario(index);
  if (!sv) {
    std::fprintf(stderr, "error: %s\n", sv.error().c_str());
    return 1;
  }

  sim::SimConfig cfg;
  cfg.agent_mode = sim::agentModeFromString(args.str("mode", "log_replay"));
  cfg.planner.backend = plan::backendFromString(args.str("backend", "ilqr"));
  cfg.pure_pursuit_only = args.has("pure-pursuit");
  cfg.replay_logged_ego = args.has("logged-ego");
  cfg.planner.cost_ctx.capabilities = shard->capabilities();
  cfg.max_steps = static_cast<std::size_t>(args.integer("max-steps", 0));
  const std::string pred = args.str("prediction", "constant_velocity");
  cfg.planner.prediction = pred == "lane_follow"     ? predict::Model::kLaneFollow
                           : pred == "logged_oracle" ? predict::Model::kLoggedOracle
                                                     : predict::Model::kConstantVelocity;
  const std::string weight = args.str("weight");
  if (!weight.empty()) {
    const auto eq = weight.find('=');
    if (eq == std::string::npos ||
        !cfg.planner.weights.setByName(weight.substr(0, eq), std::stod(weight.substr(eq + 1)))) {
      std::fprintf(stderr, "bad --weight '%s'\n", weight.c_str());
      return 2;
    }
  }

  if (args.has("dump")) {
    cfg.dump_plan_stride = static_cast<std::size_t>(args.integer("dump-plan-stride", 5));
  }

  sim::Simulator simulator;
  simulator.setup(cfg);

  // --trace replays the scenario one cycle at a time and prints what the
  // planner chose and why. This is the tool for answering "the ego is slow,
  // which cost term is responsible", which is otherwise guesswork.
  if (args.has("trace")) {
    sim::SimConfig tcfg = cfg;
    sim::Simulator tsim;
    const long n = args.integer("trace", 12);
    std::printf("%4s %6s %6s %6s %5s %5s %4s/%4s", "step", "v", "s", "d", "tgt_v", "tgt_d",
                "feas", "cand");
    for (std::size_t i = 0; i < plan::kNumCostTerms; ++i) {
      std::printf(" %8s", plan::toString(static_cast<plan::CostTerm>(i)));
    }
    std::printf("\n");
    for (long step = 0; step < n; ++step) {
      tcfg.max_steps = static_cast<std::size_t>(step) + 2;
      tsim.setup(tcfg);
      const sim::SimOutput& to = tsim.run(*sv, shard->capabilities());
      if (to.cycles.empty()) break;
      const auto& c = to.cycles.back();
      const auto& po = tsim.planner().plan(c.ego, c.a_lon, c.step);
      if (po.chosen_index < 0) break;
      const auto ci = static_cast<std::size_t>(po.chosen_index);
      std::printf("%4zu %6.2f %6.1f %6.2f %5.1f %5.1f %4zu/%4zu", c.step, c.ego.v, c.s, c.d,
                  tsim.planner().lattice()[ci].target_v, tsim.planner().lattice()[ci].target_d,
                  po.n_feasible, po.n_candidates);
      for (std::size_t i = 0; i < plan::kNumCostTerms; ++i) {
        std::printf(" %8.3f",
                    po.cost.terms[i] * tcfg.planner.weights.w[i]);
      }
      std::printf("\n");
    }
    return 0;
  }

  const sim::SimOutput& out = simulator.run(*sv, shard->capabilities());

  if (!args.has("quiet")) {
    const auto& m = out.metrics;
    std::printf("scenario        %s\n", sv->id().c_str());
    std::printf("build           %s %s %s\n", kBuildType, kCompiler, kArchFlag);
    std::printf("mode            %s   planner %s   prediction %s\n",
                sim::toString(cfg.agent_mode),
                cfg.replay_logged_ego ? "logged_ego"
              : cfg.pure_pursuit_only ? "pure_pursuit" : plan::toString(cfg.planner.backend),
                predict::toString(cfg.planner.prediction));
    std::printf("status          %s %s\n", m.status, out.failure);
    if (out.route.ok()) {
      std::printf("route           %zu lanes, %zu A* expansions, cost %.2f s\n",
                  out.route.lanes.size(), out.route.expansions, out.route.cost);
      std::printf("reference path  %.1f m at %.2f m spacing\n", simulator.path().length(),
                  simulator.path().step());
    }
    if (!out.ok) return 1;
    std::printf("cycles          %d\n", m.n_cycles);
    std::printf("collision       %d (at fault %d)\n", m.collision, m.at_fault_collision);
    std::printf("offroad         %d  max %.2f m\n", m.drivable_area_violation,
                m.max_offroad_dist);
    std::printf("wrong direction %d\n", m.wrong_direction);
    std::printf("min TTC         %.2f s\n", m.min_ttc);
    std::printf("progress ratio  %.3f   route completion %.3f\n", m.progress_ratio,
                m.route_completion);
    std::printf("comfort         a_lon %.2f  a_lat %.2f  jerk %.2f  yaw %.2f  violation %d\n",
                m.max_abs_a_lon, m.max_abs_a_lat, m.max_abs_jerk, m.max_abs_yaw_rate,
                m.comfort_violation);
    std::printf("similarity      ADE %.2f m  FDE %.2f m   (similarity, not correctness)\n", m.ade,
                m.fde);
    std::printf("plan latency    p50 %.0f us  p99 %.0f us  max %.0f us\n", m.plan_us_p50,
                m.plan_us_p99, m.plan_us_max);
    std::printf("refinement      p50 %.0f us  mean iters %.1f  converged %.0f%%\n",
                m.refine_us_p50, m.refine_iters_mean, 100.0 * m.refine_converged_frac);
  }

  const std::string dump = args.str("dump");
  if (!dump.empty()) {
    dumpJson(dump, *sv, shard->capabilities(), simulator, out,
             args.integer("dump-plan-stride", 5));
    if (!args.has("quiet")) std::printf("dumped          %s\n", dump.c_str());
  }
  return out.ok ? 0 : 1;
}
