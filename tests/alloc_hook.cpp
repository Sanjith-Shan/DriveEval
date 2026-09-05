// Replaces global operator new so that tests can assert a planning cycle
// performs no heap allocation. Linked only into sb_tests and sb_bench, never
// into the library, so production builds pay nothing for it.
#include <cstdlib>
#include <new>

#include "switchback/core/alloc_count.hpp"

void* operator new(std::size_t n) {
  sb::g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  sb::g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return operator new(n, t);
}
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
