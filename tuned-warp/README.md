# Point-only warp geometry tuning

Setup creates synthetic operands from fixed seeds and the public point. It
compares eight four/eight-warp transform geometries with the retained library
pipeline. Full output comparison precedes paired whole-graph timing. A custom
path must beat a 2% margin and twice the observed A/A spread. Otherwise the
library path remains selected.

Every scored call copies fresh actual operands and recomputes all products.
Calibration outputs are discarded. All arithmetic and completion remain in
`run`; the transfer hooks still copy raw bytes only. Setup timings are diagnostic
GPU graph intervals, not official scored results.
