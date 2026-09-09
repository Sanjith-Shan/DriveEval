// Replay many scenarios and write one metrics row and one features row each.
//
// One Simulator per worker thread, each with its own planner workspace, so the
// threads share only the mmap'd cache (read-only) and a mutex around the output
// files. Scenarios are handed out from an atomic counter rather than
// partitioned up front, because per-scenario cost varies by an order of
// magnitude with agent count and a static split leaves threads idle.
#include <atomic>
#include <ctime>
#include <filesystem>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "switchback/build_info.hpp"
#include "switchback/eval/features.hpp"
#include "switchback/sim/simulator.hpp"

using namespace sb;

namespace {

struct Counters {
  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> done{0};
  std::atomic<std::size_t> ok{0};
  std::atomic<std::size_t> no_route{0};
  std::atomic<std::size_t> planner_fail{0};
  std::atomic<std::size_t> load_fail{0};
  std::atomic<std::size_t> collisions{0};
  std::atomic<std::size_t> at_fault{0};
};

std::string isoNow() {
  const auto t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::gmtime(&t));
  return buf;
}

std::string escapeCsv(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"') out += "\"\"";
    else out += c;
  }
  out += "\"";
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  cli::Args args(argc, argv);
  args.requireNoUnknown({"shards", "out", "run-id", "config-name", "mode", "backend", "prediction",
                         "jobs", "limit", "weight", "pure-pursuit", "logged-ego", "dataset", "hardware",
                         "git-sha", "max-steps", "features", "quiet", "shard-limit",
                         "count-allocs", "suppress-agent"});

  const std::string shards_glob = args.str("shards");
  const std::string out_dir = args.str("out", "data/results");
  if (shards_glob.empty()) {
    std::fprintf(stderr,
                 "usage: sb_batch --shards 'data/cache/av2_val/*.sbsc' --out data/results\n"
                 "                [--mode log_replay|reactive] [--backend ilqr|osqp|none]\n"
                 "                [--jobs N] [--limit N] [--run-id ID] [--config-name NAME]\n"
                 "                [--weight name=value ...] [--features]\n");
    return 2;
  }

  // Expand the glob with the shell's own globbing left to the caller: the
  // argument may be a directory, a single file, or a list.
  std::vector<std::string> shard_paths;
  {
    std::string cur;
    for (const char c : shards_glob) {
      if (c == ',') {
        if (!cur.empty()) shard_paths.push_back(cur);
        cur.clear();
      } else {
        cur += c;
      }
    }
    if (!cur.empty()) shard_paths.push_back(cur);
  }
  for (const std::string& p : args.positional()) shard_paths.push_back(p);
  const auto shard_limit = static_cast<std::size_t>(args.integer("shard-limit", -1));
  if (shard_paths.size() > shard_limit) shard_paths.resize(shard_limit);

  sim::SimConfig cfg;
  cfg.agent_mode = sim::agentModeFromString(args.str("mode", "log_replay"));
  cfg.planner.backend = plan::backendFromString(args.str("backend", "ilqr"));
  cfg.pure_pursuit_only = args.has("pure-pursuit");
  cfg.replay_logged_ego = args.has("logged-ego");
  cfg.max_steps = static_cast<std::size_t>(args.integer("max-steps", 0));
  cfg.count_allocs = args.has("count-allocs");
  cfg.suppress_agent = static_cast<int>(args.integer("suppress-agent", -1));
  const std::string pred = args.str("prediction", "constant_velocity");
  cfg.planner.prediction = pred == "lane_follow"     ? predict::Model::kLaneFollow
                           : pred == "logged_oracle" ? predict::Model::kLoggedOracle
                                                     : predict::Model::kConstantVelocity;
  // --weight may repeat; cli::Args keeps the last, so a sweep passes one.
  const std::string weight = args.str("weight");
  if (!weight.empty()) {
    const auto eq = weight.find('=');
    if (eq == std::string::npos ||
        !cfg.planner.weights.setByName(weight.substr(0, eq), std::stod(weight.substr(eq + 1)))) {
      std::fprintf(stderr, "bad --weight '%s'\n", weight.c_str());
      return 2;
    }
  }

  const std::string dataset = args.str("dataset", "av2_val");
  const std::string run_id = args.str("run-id", "run_" + std::to_string(std::time(nullptr)));
  const std::string config_name = args.str("config-name", "baseline");
  const std::string hardware =
      args.str("hardware", "Apple M3 Pro, " + std::string(kArchFlag) + ", process not pinned");
  const bool want_features = args.has("features");
  const auto limit = static_cast<std::size_t>(args.integer("limit", -1));
  auto jobs = static_cast<unsigned>(args.integer("jobs", 0));
  if (jobs == 0) jobs = std::max(1U, std::thread::hardware_concurrency());

  // Open every shard up front. The mappings are read-only and shared across
  // threads, which is why a batch over 3.3 GB of cache uses almost no RSS.
  std::vector<io::ShardReader> shards;
  std::vector<std::pair<std::size_t, std::size_t>> work;  // (shard, index)
  io::u32 capabilities = 0;
  for (const std::string& p : shard_paths) {
    auto r = io::ShardReader::open(p);
    if (!r) {
      std::fprintf(stderr, "warning: %s\n", r.error().c_str());
      continue;
    }
    capabilities = r->capabilities();
    const std::size_t si = shards.size();
    for (std::size_t i = 0; i < r->size(); ++i) work.emplace_back(si, i);
    shards.push_back(std::move(*r));
  }
  if (work.empty()) {
    std::fprintf(stderr, "no scenarios found\n");
    return 1;
  }
  if (work.size() > limit) work.resize(limit);
  cfg.planner.cost_ctx.capabilities = capabilities;

  std::filesystem::create_directories(out_dir);
  const std::string metrics_path = out_dir + "/metrics_" + run_id + ".csv";
  const std::string features_path = out_dir + "/features_" + dataset + ".csv";
  const std::string runs_path = out_dir + "/runs.csv";

  std::FILE* fm = std::fopen(metrics_path.c_str(), "w");
  std::FILE* ff = want_features ? std::fopen(features_path.c_str(), "w") : nullptr;
  if (fm == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", metrics_path.c_str());
    return 1;
  }
  std::fprintf(fm, "%s\n", eval::metricsCsvHeader().c_str());
  if (ff != nullptr) std::fprintf(ff, "%s\n", eval::featuresCsvHeader().c_str());

  Counters counters;
  std::mutex out_mutex;
  const auto t_start = std::chrono::steady_clock::now();
  const bool quiet = args.has("quiet");

  auto worker = [&]() {
    sim::Simulator simulator;
    simulator.setup(cfg);
    std::string buf_m, buf_f;
    buf_m.reserve(1 << 16);
    buf_f.reserve(1 << 16);
    std::size_t local = 0;

    while (true) {
      const std::size_t idx = counters.next.fetch_add(1, std::memory_order_relaxed);
      if (idx >= work.size()) break;
      const auto [si, sc] = work[idx];
      auto sv = shards[si].scenario(sc);
      if (!sv) {
        counters.load_fail.fetch_add(1, std::memory_order_relaxed);
        counters.done.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      const sim::SimOutput& so = simulator.run(*sv, capabilities);

      eval::ScenarioMetrics m = so.metrics;
      m.scenario_id = sv->id();
      buf_m += eval::metricsCsvRow(run_id, m);
      buf_m += "\n";

      if (std::string(m.status) == "ok") {
        counters.ok.fetch_add(1, std::memory_order_relaxed);
        if (m.collision != 0) counters.collisions.fetch_add(1, std::memory_order_relaxed);
        if (m.at_fault_collision != 0) counters.at_fault.fetch_add(1, std::memory_order_relaxed);
      } else if (std::string(m.status) == "no_route") {
        counters.no_route.fetch_add(1, std::memory_order_relaxed);
      } else if (std::string(m.status) == "planner_fail") {
        counters.planner_fail.fetch_add(1, std::memory_order_relaxed);
      } else {
        counters.load_fail.fetch_add(1, std::memory_order_relaxed);
      }

      // Features describe the situation and are planner-independent, so they
      // are written only when asked for and only once per dataset.
      if (ff != nullptr && so.route.ok()) {
        const eval::ScenarioFeatures fe = eval::extractFeatures(
            *sv, simulator.graph(), so.route, simulator.path(), capabilities, dataset);
        buf_f += eval::featuresCsvRow(fe);
        buf_f += "\n";
      }

      if (++local >= 64) {
        const std::lock_guard<std::mutex> lock(out_mutex);
        std::fwrite(buf_m.data(), 1, buf_m.size(), fm);
        if (ff != nullptr) std::fwrite(buf_f.data(), 1, buf_f.size(), ff);
        buf_m.clear();
        buf_f.clear();
        local = 0;
      }
      const std::size_t d = counters.done.fetch_add(1, std::memory_order_relaxed) + 1;
      if (!quiet && d % 500 == 0) {
        const auto el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
                            .count();
        std::fprintf(stderr, "\r  %zu/%zu  %.0f scen/s  %.0f s elapsed   ", d, work.size(),
                     static_cast<double>(d) / el, el);
        std::fflush(stderr);
      }
    }
    const std::lock_guard<std::mutex> lock(out_mutex);
    std::fwrite(buf_m.data(), 1, buf_m.size(), fm);
    if (ff != nullptr) std::fwrite(buf_f.data(), 1, buf_f.size(), ff);
  };

  std::vector<std::thread> pool;
  pool.reserve(jobs);
  for (unsigned i = 0; i < jobs; ++i) pool.emplace_back(worker);
  for (std::thread& t : pool) t.join();
  std::fclose(fm);
  if (ff != nullptr) std::fclose(ff);
  const auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

  // The runs table carries the provenance every number in the report needs.
  const bool runs_exists = std::filesystem::exists(runs_path);
  std::FILE* fr = std::fopen(runs_path.c_str(), runs_exists ? "a" : "w");
  if (fr != nullptr) {
    if (!runs_exists) {
      std::fprintf(fr,
                   "run_id,created_at,config_name,config_json,agent_mode,planner,dataset,git_sha,"
                   "hardware,capabilities,n_scenarios\n");
    }
    std::fprintf(fr, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%u,%zu\n", run_id.c_str(), isoNow().c_str(),
                 config_name.c_str(), escapeCsv(cfg.planner.toJson()).c_str(),
                 sim::toString(cfg.agent_mode),
                 cfg.replay_logged_ego ? "logged_ego"
              : cfg.pure_pursuit_only ? "pure_pursuit" : plan::toString(cfg.planner.backend),
                 dataset.c_str(), args.str("git-sha", "unknown").c_str(),
                 escapeCsv(hardware).c_str(), capabilities, work.size());
    std::fclose(fr);
  }

  std::fprintf(stderr, "\r%60s\r", "");
  std::printf("run             %s (%s)\n", run_id.c_str(), config_name.c_str());
  std::printf("build           %s %s %s\n", kBuildType, kCompiler, kArchFlag);
  std::printf("hardware        %s\n", hardware.c_str());
  std::printf("mode            %s   planner %s   prediction %s\n", sim::toString(cfg.agent_mode),
              cfg.replay_logged_ego ? "logged_ego"
              : cfg.pure_pursuit_only ? "pure_pursuit" : plan::toString(cfg.planner.backend),
              predict::toString(cfg.planner.prediction));
  std::printf("scenarios       %zu over %zu shards, %u threads\n", work.size(), shards.size(),
              jobs);
  std::printf("  ok            %zu\n", counters.ok.load());
  std::printf("  no_route      %zu\n", counters.no_route.load());
  std::printf("  planner_fail  %zu\n", counters.planner_fail.load());
  std::printf("  load_fail     %zu\n", counters.load_fail.load());
  std::printf("collisions      %zu (at fault %zu)\n", counters.collisions.load(),
              counters.at_fault.load());
  std::printf("wall            %.1f s   throughput %.0f scenarios/s\n", wall,
              static_cast<double>(work.size()) / wall);
  std::printf("metrics         %s\n", metrics_path.c_str());
  if (ff != nullptr) std::printf("features        %s\n", features_path.c_str());
  return 0;
}
