#ifndef SWITCHBACK_CORE_TYPES_HPP
#define SWITCHBACK_CORE_TYPES_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace sb {

// The planner runs in double throughout. The cache stores float32 to halve its
// footprint, but every derived quantity -- curvature, jerk, yaw rate -- is a
// second or third difference, and float32 differencing at 0.1 s spacing loses
// more precision than the memory saving is worth.
using Scalar = double;

inline constexpr Scalar kPi = std::numbers::pi_v<Scalar>;
inline constexpr Scalar kEps = 1e-9;

struct Vec2 {
  Scalar x{0.0};
  Scalar y{0.0};

  constexpr Vec2() = default;
  constexpr Vec2(Scalar xx, Scalar yy) : x(xx), y(yy) {}

  constexpr Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
  constexpr Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
  constexpr Vec2 operator*(Scalar s) const { return {x * s, y * s}; }
  constexpr Vec2 operator/(Scalar s) const { return {x / s, y / s}; }
  constexpr Vec2 operator-() const { return {-x, -y}; }
  Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
  Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
  Vec2& operator*=(Scalar s) { x *= s; y *= s; return *this; }

  [[nodiscard]] constexpr Scalar dot(const Vec2& o) const { return x * o.x + y * o.y; }
  [[nodiscard]] constexpr Scalar cross(const Vec2& o) const { return x * o.y - y * o.x; }
  [[nodiscard]] constexpr Scalar squaredNorm() const { return x * x + y * y; }
  [[nodiscard]] Scalar norm() const { return std::sqrt(squaredNorm()); }
  [[nodiscard]] Vec2 normalized() const {
    const Scalar n = norm();
    return n > kEps ? Vec2{x / n, y / n} : Vec2{0.0, 0.0};
  }
  // Unit normal rotated +90 degrees, which is the left-hand side of travel.
  [[nodiscard]] constexpr Vec2 perp() const { return {-y, x}; }
  [[nodiscard]] Scalar angle() const { return std::atan2(y, x); }
  [[nodiscard]] Vec2 rotated(Scalar theta) const {
    const Scalar c = std::cos(theta);
    const Scalar s = std::sin(theta);
    return {x * c - y * s, x * s + y * c};
  }
};

inline constexpr Vec2 operator*(Scalar s, const Vec2& v) { return v * s; }
inline Scalar distance(const Vec2& a, const Vec2& b) { return (a - b).norm(); }
inline Scalar squaredDistance(const Vec2& a, const Vec2& b) { return (a - b).squaredNorm(); }

// Wrap to (-pi, pi]. Every heading comparison in the codebase goes through this
// or through angleDiff; comparing raw headings is the bug that makes a planner
// mysteriously refuse to turn left across the +pi seam.
inline Scalar wrapAngle(Scalar a) {
  a = std::fmod(a + kPi, 2.0 * kPi);
  if (a <= 0.0) a += 2.0 * kPi;
  return a - kPi;
}

inline Scalar angleDiff(Scalar a, Scalar b) { return wrapAngle(a - b); }

inline Scalar lerpAngle(Scalar a, Scalar b, Scalar t) { return wrapAngle(a + angleDiff(b, a) * t); }

template <typename T>
constexpr T clampT(T v, T lo, T hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

struct Pose2 {
  Scalar x{0.0};
  Scalar y{0.0};
  Scalar theta{0.0};
  [[nodiscard]] Vec2 position() const { return {x, y}; }
};

// Kinematic bicycle state, rear-axle referenced. v may be negative, though the
// planner never commands reverse.
struct EgoState {
  Scalar x{0.0};
  Scalar y{0.0};
  Scalar theta{0.0};
  Scalar v{0.0};
  [[nodiscard]] Vec2 position() const { return {x, y}; }
};

struct Control {
  Scalar a{0.0};      // longitudinal acceleration, m/s^2
  Scalar delta{0.0};  // front wheel angle, radians
};

// Point in a Frenet frame attached to a reference path.
struct FrenetPoint {
  Scalar s{0.0};   // arc length along the reference
  Scalar d{0.0};   // signed lateral offset, positive to the left
  Scalar ds{0.0};  // ds/dt
  Scalar dd{0.0};  // dd/dt
};

// Oriented bounding box. Every footprint in the harness is one of these, which
// is why collision checking has exactly one implementation to get right.
struct Box2 {
  Vec2 center{};
  Scalar theta{0.0};
  Scalar length{0.0};  // along heading
  Scalar width{0.0};   // across heading

  [[nodiscard]] Scalar circumradius() const { return 0.5 * std::hypot(length, width); }
};

// Vehicle geometry and limits. Defaults are a mid-size sedan; the reference
// point is the rear axle and the footprint is centred on the geometric centre,
// which is rear_axle_to_center ahead of it.
struct VehicleParams {
  Scalar length{4.8};
  Scalar width{2.0};
  Scalar wheelbase{2.9};
  Scalar rear_axle_to_center{1.45};
  Scalar max_steer{0.60};          // rad, roughly 34 degrees
  Scalar max_steer_rate{0.50};     // rad/s
  Scalar max_accel{3.0};           // m/s^2
  Scalar max_decel{-6.0};          // m/s^2, comfortable-hard braking
  Scalar max_speed{25.0};          // m/s
  [[nodiscard]] Scalar maxCurvature() const { return std::tan(max_steer) / wheelbase; }
};

// Every trajectory sample in this codebase is the vehicle's REAR AXLE, because
// that is the reference point of the kinematic bicycle model. The footprint is
// centred rear_axle_to_center ahead of it. Mixing the two reference points is
// the classic source of a planner that collides at its own front bumper while
// reporting clearance, so the conversion exists exactly once, here, and every
// collision check goes through it.
inline Box2 egoFootprint(Scalar x, Scalar y, Scalar heading, const VehicleParams& v) {
  const Scalar off = v.rear_axle_to_center;
  return Box2{Vec2{x + std::cos(heading) * off, y + std::sin(heading) * off}, heading, v.length,
              v.width};
}

inline Box2 egoFootprint(const EgoState& s, const VehicleParams& v) {
  return egoFootprint(s.x, s.y, s.theta, v);
}

}  // namespace sb

#endif  // SWITCHBACK_CORE_TYPES_HPP
