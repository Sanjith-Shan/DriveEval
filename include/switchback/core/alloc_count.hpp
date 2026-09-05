#ifndef SWITCHBACK_CORE_ALLOC_COUNT_HPP
#define SWITCHBACK_CORE_ALLOC_COUNT_HPP

#include <atomic>
#include <cstdint>

namespace sb {

// Global heap-allocation counter. The library only reads it; the counting
// happens in a translation unit that replaces global operator new, which is
// linked into the test binary and into sb_bench. That keeps the counter free in
// production builds while letting tests/test_no_alloc.cpp assert that a
// planning cycle allocates nothing, rather than the README merely claiming it.
extern std::atomic<std::int64_t> g_alloc_count;

inline std::int64_t allocationCount() { return g_alloc_count.load(std::memory_order_relaxed); }

// Scoped delta, for wrapping a region whose allocation behaviour matters.
class AllocScope {
 public:
  AllocScope() : start_(allocationCount()) {}
  [[nodiscard]] std::int64_t delta() const { return allocationCount() - start_; }
  void reset() { start_ = allocationCount(); }

 private:
  std::int64_t start_;
};

}  // namespace sb

#endif  // SWITCHBACK_CORE_ALLOC_COUNT_HPP
