// Minimal flag parsing shared by the sb_* tools. Deliberately small: the tools
// take a handful of flags each and a real options library would be the largest
// dependency in the project.
#ifndef SWITCHBACK_APPS_CLI_HPP
#define SWITCHBACK_APPS_CLI_HPP

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sb::cli {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string_view a(argv[i]);
      if (a.starts_with("--")) {
        a.remove_prefix(2);
        const auto eq = a.find('=');
        if (eq != std::string_view::npos) {
          flags_[std::string(a.substr(0, eq))] = std::string(a.substr(eq + 1));
        } else if (i + 1 < argc && argv[i + 1][0] != '-') {
          flags_[std::string(a)] = argv[++i];
        } else {
          flags_[std::string(a)] = "1";
        }
      } else {
        positional_.emplace_back(a);
      }
    }
  }

  [[nodiscard]] bool has(const std::string& k) const { return flags_.count(k) > 0; }
  [[nodiscard]] std::string str(const std::string& k, std::string def = {}) const {
    const auto it = flags_.find(k);
    return it == flags_.end() ? std::move(def) : it->second;
  }
  [[nodiscard]] double num(const std::string& k, double def) const {
    const auto it = flags_.find(k);
    if (it == flags_.end()) return def;
    try {
      return std::stod(it->second);
    } catch (...) {
      std::cerr << "switchback: --" << k << " expects a number, got '" << it->second << "'\n";
      std::exit(2);
    }
  }
  [[nodiscard]] long integer(const std::string& k, long def) const {
    return static_cast<long>(num(k, static_cast<double>(def)));
  }
  [[nodiscard]] const std::vector<std::string>& positional() const { return positional_; }

  // Any flag the tool never asked about is almost certainly a typo, and
  // silently ignoring it is how a benchmark ends up run with default settings
  // while its operator believes otherwise.
  void requireNoUnknown(const std::vector<std::string>& known) const {
    for (const auto& [k, v] : flags_) {
      (void)v;
      bool found = false;
      for (const auto& kk : known) {
        if (k == kk) { found = true; break; }
      }
      if (!found) {
        std::cerr << "switchback: unknown flag --" << k << "\n";
        std::exit(2);
      }
    }
  }

 private:
  std::map<std::string, std::string> flags_;
  std::vector<std::string> positional_;
};

}  // namespace sb::cli

#endif
