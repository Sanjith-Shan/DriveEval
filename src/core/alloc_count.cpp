#include "switchback/core/alloc_count.hpp"

namespace sb {
std::atomic<std::int64_t> g_alloc_count{0};
}  // namespace sb
