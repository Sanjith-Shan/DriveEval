#include "driveeval/core/alloc_count.hpp"

namespace drive {
std::atomic<std::int64_t> g_alloc_count{0};
}  // namespace drive
