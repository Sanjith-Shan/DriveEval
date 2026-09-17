#include "driveeval/map/drivable_area.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace drive::map {

DrivableArea DrivableArea::build(const io::ScenarioView& view, Vec2 focus_min,
                                 Vec2 focus_max) {
  DrivableArea out;
  for (std::size_t i = 0; i < view.polygons.size(); ++i) {
    if (view.polygons[i].kind != io::kPolyDrivableArea) continue;
    const auto pts = view.polygonPoints(i);
    if (pts.size() < 3) continue;
    std::vector<Vec2> poly;
    poly.reserve(pts.size());
    Bounds b{std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max(),
             std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::lowest()};
    for (const io::PointRec& p : pts) {
      const Vec2 v = view.point(p);
      poly.push_back(v);
      b.minx = std::min(b.minx, v.x);
      b.miny = std::min(b.miny, v.y);
      b.maxx = std::max(b.maxx, v.x);
      b.maxy = std::max(b.maxy, v.y);
    }
    out.polys_.push_back(std::move(poly));
    out.bounds_.push_back(b);
  }
  // 0.25 m cells with a 12 m margin. Measured against the exact polygon scan
  // over 160,000 random points, this field's residual error is reported in
  // docs/LIMITATIONS.md; the cost function uses it, while the reported off-road
  // metric uses the exact scan, so no published number depends on the
  // approximation.
  out.buildField(0.25, 12.0, focus_min, focus_max);
  return out;
}

bool DrivableArea::contains(const Vec2& p) const {
  for (std::size_t i = 0; i < polys_.size(); ++i) {
    if (!bounds_[i].maybeContains(p)) continue;
    if (pointInPolygon(p, polys_[i])) return true;
  }
  return false;
}

namespace {

// One-dimensional exact squared Euclidean distance transform, Felzenszwalb and
// Huttenlocher's lower-envelope algorithm. Linear in the row length, and exact,
// which matters because a chamfer approximation would put a systematic bias
// into the off-road metric.
void edt1d(const float* f, float* d, int n, int* v, float* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -1e20F;
  z[1] = 1e20F;
  for (int q = 1; q < n; ++q) {
    float s = 0.0F;
    while (true) {
      const int p = v[k];
      s = ((f[q] + static_cast<float>(q) * static_cast<float>(q)) -
           (f[p] + static_cast<float>(p) * static_cast<float>(p))) /
          (2.0F * static_cast<float>(q) - 2.0F * static_cast<float>(p));
      if (s > z[k]) break;
      --k;
      if (k < 0) {
        k = 0;
        break;
      }
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e20F;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < static_cast<float>(q)) ++k;
    const float dq = static_cast<float>(q) - static_cast<float>(v[k]);
    d[q] = dq * dq + f[v[k]];
  }
}

}  // namespace

void DrivableArea::buildField(Scalar resolution, Scalar margin, Vec2 focus_min, Vec2 focus_max) {
  if (polys_.empty()) return;
  Scalar maxx = std::numeric_limits<Scalar>::lowest();
  Scalar maxy = std::numeric_limits<Scalar>::lowest();
  minx_ = std::numeric_limits<Scalar>::max();
  miny_ = std::numeric_limits<Scalar>::max();
  for (const Bounds& b : bounds_) {
    minx_ = std::min(minx_, b.minx);
    miny_ = std::min(miny_, b.miny);
    maxx = std::max(maxx, b.maxx);
    maxy = std::max(maxy, b.maxy);
  }
  minx_ -= margin;
  miny_ -= margin;
  maxx += margin;
  maxy += margin;
  // Intersect with the caller's region of interest, when given one.
  if (focus_max.x > focus_min.x && focus_max.y > focus_min.y) {
    minx_ = std::max(minx_, focus_min.x - margin);
    miny_ = std::max(miny_, focus_min.y - margin);
    maxx = std::min(maxx, focus_max.x + margin);
    maxy = std::min(maxy, focus_max.y + margin);
    if (maxx <= minx_ || maxy <= miny_) {
      dist_.clear();
      nx_ = ny_ = 0;
      return;
    }
  }
  res_ = resolution;
  nx_ = static_cast<int>(std::ceil((maxx - minx_) / res_)) + 1;
  ny_ = static_cast<int>(std::ceil((maxy - miny_) / res_)) + 1;
  // A map large enough to blow the memory budget falls back to the exact path
  // rather than silently allocating hundreds of megabytes.
  if (nx_ <= 0 || ny_ <= 0 || static_cast<long long>(nx_) * ny_ > 16'000'000LL) {
    dist_.clear();
    nx_ = ny_ = 0;
    return;
  }

  const auto cells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
  std::vector<float> f(cells, 1e20F);

  // Scanline fill of the polygon union. Per row this is O(edges) rather than
  // the O(cells * edges) a point-in-polygon test per cell would cost.
  std::vector<Scalar> xs;
  for (int j = 0; j < ny_; ++j) {
    const Scalar y = miny_ + static_cast<Scalar>(j) * res_;
    for (const std::vector<Vec2>& poly : polys_) {
      xs.clear();
      for (std::size_t a = 0, b = poly.size() - 1; a < poly.size(); b = a++) {
        const Vec2& p1 = poly[b];
        const Vec2& p2 = poly[a];
        // Same half-open rule as pointInPolygon, so the raster and the exact
        // test agree on boundary cases instead of disagreeing by one row.
        if ((p2.y > y) != (p1.y > y)) {
          xs.push_back(p2.x + (y - p2.y) / (p1.y - p2.y) * (p1.x - p2.x));
        }
      }
      std::sort(xs.begin(), xs.end());
      for (std::size_t k = 0; k + 1 < xs.size(); k += 2) {
        int i0 = static_cast<int>(std::ceil((xs[k] - minx_) / res_));
        int i1 = static_cast<int>(std::floor((xs[k + 1] - minx_) / res_));
        i0 = std::max(i0, 0);
        i1 = std::min(i1, nx_ - 1);
        for (int i = i0; i <= i1; ++i) {
          f[static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) +
            static_cast<std::size_t>(i)] = 0.0F;
        }
      }
    }
  }

  // Exact EDT: transform along columns, then along rows.
  const int maxdim = std::max(nx_, ny_);
  std::vector<float> buf_d(static_cast<std::size_t>(maxdim));
  std::vector<int> v(static_cast<std::size_t>(maxdim));
  std::vector<float> z(static_cast<std::size_t>(maxdim) + 1);

  // Transpose once and walk rows, rather than striding down columns of a wide
  // grid and taking a cache miss per cell.
  std::vector<float> t(cells);
  for (int j = 0; j < ny_; ++j) {
    for (int i = 0; i < nx_; ++i) {
      t[static_cast<std::size_t>(i) * static_cast<std::size_t>(ny_) +
        static_cast<std::size_t>(j)] =
          f[static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) +
            static_cast<std::size_t>(i)];
    }
  }
  for (int i = 0; i < nx_; ++i) {
    float* col = &t[static_cast<std::size_t>(i) * static_cast<std::size_t>(ny_)];
    edt1d(col, buf_d.data(), ny_, v.data(), z.data());
    for (int j = 0; j < ny_; ++j) col[j] = buf_d[static_cast<std::size_t>(j)];
  }
  for (int j = 0; j < ny_; ++j) {
    for (int i = 0; i < nx_; ++i) {
      f[static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) +
        static_cast<std::size_t>(i)] =
          t[static_cast<std::size_t>(i) * static_cast<std::size_t>(ny_) +
            static_cast<std::size_t>(j)];
    }
  }
  for (int j = 0; j < ny_; ++j) {
    float* row = &f[static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_)];
    edt1d(row, buf_d.data(), nx_, v.data(), z.data());
    for (int i = 0; i < nx_; ++i) row[i] = buf_d[static_cast<std::size_t>(i)];
  }

  dist_.resize(cells);
  // The transform measures the distance to the nearest cell whose centre is
  // inside, and that centre sits up to half a cell inside the true boundary, so
  // the raw value is biased outward by a systematic half cell. Removing it and
  // clamping at zero leaves only the quantisation spread.
  const double bias = 0.5 * static_cast<double>(res_);
  for (std::size_t k = 0; k < cells; ++k) {
    const double d = std::sqrt(static_cast<double>(f[k])) * static_cast<double>(res_);
    dist_[k] = static_cast<float>(d > bias ? d - bias : 0.0);
  }
}

Scalar DrivableArea::outsideDistanceExact(const Vec2& p) const {
  // A map with no drivable polygons cannot answer the question. Returning zero
  // would silently mark every pose as on-road, so the metric layer checks the
  // capability bit before calling this at all.
  if (polys_.empty()) return 0.0;
  Scalar best = std::numeric_limits<Scalar>::max();
  for (std::size_t i = 0; i < polys_.size(); ++i) {
    const Scalar d = pointOutsidePolygonDistance(p, polys_[i]);
    if (d == 0.0) return 0.0;
    best = std::min(best, d);
  }
  return best;
}

Scalar DrivableArea::outsideDistance(const Vec2& p) const {
  if (dist_.empty()) return outsideDistanceExact(p);
  const Scalar gx = (p.x - minx_) / res_;
  const Scalar gy = (p.y - miny_) / res_;
  const int i = static_cast<int>(std::floor(gx));
  const int j = static_cast<int>(std::floor(gy));
  // Outside the field entirely. Rare, and answered exactly rather than
  // extrapolated, because the field's margin is not a bound on where a
  // diverging planner can put the ego.
  if (i < 0 || j < 0 || i + 1 >= nx_ || j + 1 >= ny_) return outsideDistanceExact(p);
  // Bilinear rather than nearest-cell. Nearest-cell alone quantises the query
  // to half a cell diagonal, which was most of the residual error against the
  // exact scan.
  const auto idx = [&](int a, int b) {
    return static_cast<std::size_t>(b) * static_cast<std::size_t>(nx_) +
           static_cast<std::size_t>(a);
  };
  const Scalar tx = gx - static_cast<Scalar>(i);
  const Scalar ty = gy - static_cast<Scalar>(j);
  const Scalar d00 = static_cast<Scalar>(dist_[idx(i, j)]);
  const Scalar d10 = static_cast<Scalar>(dist_[idx(i + 1, j)]);
  const Scalar d01 = static_cast<Scalar>(dist_[idx(i, j + 1)]);
  const Scalar d11 = static_cast<Scalar>(dist_[idx(i + 1, j + 1)]);
  const Scalar top = d00 * (1.0 - tx) + d10 * tx;
  const Scalar bot = d01 * (1.0 - tx) + d11 * tx;
  return top * (1.0 - ty) + bot * ty;
}

Scalar DrivableArea::outsideDistance(const Box2& box) const {
  if (polys_.empty()) return 0.0;
  const BoxCorners c = corners(box);
  Scalar worst = 0.0;
  for (const Vec2& p : c) worst = std::max(worst, outsideDistance(p));
  return worst;
}

}  // namespace drive::map
