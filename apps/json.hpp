// Minimal JSON writer. The trajectory dump is the only JSON this project
// produces and its shape is fixed by docs/CONTRACTS.md, so a streaming writer
// with no DOM is enough and keeps the dependency list short.
#ifndef SWITCHBACK_APPS_JSON_HPP
#define SWITCHBACK_APPS_JSON_HPP

#include <cstdio>
#include <string>
#include <vector>

namespace sb::json {

class Writer {
 public:
  explicit Writer(std::FILE* f) : f_(f) {}

  void beginObject() { punct('{'); stack_.push_back(false); }
  void endObject() { stack_.pop_back(); std::fputc('}', f_); after(); }
  void beginArray() { punct('['); stack_.push_back(false); }
  void endArray() { stack_.pop_back(); std::fputc(']', f_); after(); }

  void key(const char* k) {
    comma();
    std::fprintf(f_, "\"%s\":", k);
    suppress_comma_ = true;
  }

  void value(const char* s) { comma(); std::fprintf(f_, "\"%s\"", s); after(); }
  void value(const std::string& s) { value(s.c_str()); }
  void value(double d) {
    comma();
    // %.6g keeps the dump compact while preserving centimetre precision on
    // coordinates, which is well inside the dataset's own accuracy.
    if (d != d) {
      std::fputs("null", f_);
    } else {
      std::fprintf(f_, "%.6g", d);
    }
    after();
  }
  void value(long v) { comma(); std::fprintf(f_, "%ld", v); after(); }
  void value(int v) { value(static_cast<long>(v)); }
  void value(unsigned long v) { comma(); std::fprintf(f_, "%lu", v); after(); }
  void value(bool b) { comma(); std::fputs(b ? "true" : "false", f_); after(); }

  void kv(const char* k, double v) { key(k); value(v); }
  void kv(const char* k, long v) { key(k); value(v); }
  void kv(const char* k, int v) { key(k); value(v); }
  void kv(const char* k, unsigned long v) { key(k); value(v); }
  void kv(const char* k, bool v) { key(k); value(v); }
  void kv(const char* k, const char* v) { key(k); value(v); }
  void kv(const char* k, const std::string& v) { key(k); value(v); }
  // Emits an already-formatted JSON fragment, for config blobs built elsewhere.
  void kvRaw(const char* k, const std::string& v) {
    key(k);
    comma();
    std::fputs(v.c_str(), f_);
    after();
  }

  void xy(double x, double y) {
    comma();
    std::fprintf(f_, "[%.4f,%.4f]", x, y);
    after();
  }

 private:
  void comma() {
    if (suppress_comma_) {
      suppress_comma_ = false;
      return;
    }
    if (!stack_.empty() && stack_.back()) std::fputc(',', f_);
  }
  void after() {
    if (!stack_.empty()) stack_.back() = true;
  }
  void punct(char c) {
    comma();
    std::fputc(c, f_);
  }

  std::FILE* f_;
  std::vector<bool> stack_;
  bool suppress_comma_{false};
};

}  // namespace sb::json

#endif
