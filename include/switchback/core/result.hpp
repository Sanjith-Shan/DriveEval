#ifndef SWITCHBACK_CORE_RESULT_HPP
#define SWITCHBACK_CORE_RESULT_HPP

#include <optional>
#include <string>
#include <utility>

namespace sb {

// A value or an explanation. Used on every boundary that can fail for reasons
// outside the program -- a truncated cache file, a scenario with no route -- so
// that those become reported outcomes rather than exceptions or crashes. The
// harness must be able to say "3 of 12,000 scenarios failed to load, here is
// why", and that is only possible if failure carries a reason.
template <typename T>
class Result {
 public:
  static Result success(T value) {
    Result r;
    r.value_ = std::move(value);
    return r;
  }
  static Result failure(std::string error) {
    Result r;
    r.error_ = std::move(error);
    return r;
  }

  [[nodiscard]] bool ok() const { return value_.has_value(); }
  explicit operator bool() const { return ok(); }
  [[nodiscard]] const std::string& error() const { return error_; }
  [[nodiscard]] T& value() { return *value_; }
  [[nodiscard]] const T& value() const { return *value_; }
  T* operator->() { return &*value_; }
  const T* operator->() const { return &*value_; }
  T& operator*() { return *value_; }
  const T& operator*() const { return *value_; }

 private:
  Result() = default;
  std::optional<T> value_;
  std::string error_;
};

}  // namespace sb

#endif  // SWITCHBACK_CORE_RESULT_HPP
