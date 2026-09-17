"""Argoverse 2 motion-forecasting ingest: download, convert, speed prior.

AV2 MF ships a lane graph and drivable area but no posted speed limits, no
traffic lights, no stop signs and no box dimensions. Every place where this
package substitutes a constant or an estimate for a measurement is marked,
because the capability bits in the cache header are what stop the metric
suite from scoring a metric the dataset cannot actually support.

Submodules are imported directly (driveeval.av2.convert, .download,
.speed_prior) rather than re-exported here, so the downloader does not drag
pyarrow in with it.
"""
