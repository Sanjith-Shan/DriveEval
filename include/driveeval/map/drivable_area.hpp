#ifndef DRIVEEVAL_MAP_DRIVABLE_AREA_HPP
#define DRIVEEVAL_MAP_DRIVABLE_AREA_HPP

#include <vector>

#include "driveeval/core/geometry.hpp"
#include "driveeval/io/cache_reader.hpp"

namespace drive::map {

// The union of the map's drivable polygons, with per-polygon bounding boxes so
// the common case -- a query well inside one polygon -- costs one box test and
// one ray cast rather than a scan of every boundary in the map.
class DrivableArea {
 public:
  // `focus_min`/`focus_max` bound the region the distance field is built over.
  // Rasterising the whole map cost 396 ms per scenario, which was 40% of the
  // total time to simulate one: an Argoverse map spans a few hundred metres in
  // each direction while the ego only ever visits its own route corridor.
  // Outside the field, queries fall back to the exact scan, so narrowing it
  // costs accuracy nowhere and only costs speed on a query the ego never makes.
  // An empty focus box means the whole map.
  static DrivableArea build(const io::ScenarioView& view, Vec2 focus_min = Vec2{1.0, 1.0},
                            Vec2 focus_max = Vec2{-1.0, -1.0});

  [[nodiscard]] bool empty() const { return polys_.empty(); }
  [[nodiscard]] std::size_t size() const { return polys_.size(); }
  [[nodiscard]] const std::vector<Vec2>& polygon(std::size_t i) const { return polys_[i]; }

  [[nodiscard]] bool contains(const Vec2& p) const;
  // Zero when inside any polygon, otherwise the distance to the nearest
  // boundary. This is the quantity the drivable-area metric reports, so that
  // "0.1 m over the edge" and "4 m into a building" are distinguishable.
  //
  // Answered from a precomputed distance field. The exact polygon scan it
  // replaces cost 3.8 us per call, and the cost function calls it four times
  // per trajectory sample for sixty candidates over fifty-one steps, which was
  // 47 ms of a 100 ms planning budget spent on one geometric query. The field
  // is built once per scenario and makes the query a grid lookup.
  [[nodiscard]] Scalar outsideDistance(const Vec2& p) const;
  // Worst outsideDistance over a footprint's four corners. A vehicle is off the
  // drivable area when any part of it is, not when its centre is.
  [[nodiscard]] Scalar outsideDistance(const Box2& box) const;

  // The exact polygon scan, kept so that tests can hold the grid to it rather
  // than trusting that a faster answer is still the right one.
  [[nodiscard]] Scalar outsideDistanceExact(const Vec2& p) const;
  [[nodiscard]] bool hasGrid() const { return !dist_.empty(); }
  [[nodiscard]] Scalar gridResolution() const { return res_; }

 private:
  // Scanline-rasterise the polygon union, then run an exact Euclidean distance
  // transform over the result.
  void buildField(Scalar resolution, Scalar margin, Vec2 focus_min, Vec2 focus_max);
  struct Bounds {
    Scalar minx{0}, miny{0}, maxx{0}, maxy{0};
    [[nodiscard]] bool maybeContains(const Vec2& p) const {
      return p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy;
    }
  };
  std::vector<std::vector<Vec2>> polys_;
  std::vector<Bounds> bounds_;

  // Distance field. dist_[j * nx_ + i] is the distance in metres from that
  // cell's centre to the nearest drivable cell, and zero inside.
  std::vector<float> dist_;
  Scalar res_{0.5};
  Scalar minx_{0.0}, miny_{0.0};
  int nx_{0}, ny_{0};
};

}  // namespace drive::map

#endif  // DRIVEEVAL_MAP_DRIVABLE_AREA_HPP
