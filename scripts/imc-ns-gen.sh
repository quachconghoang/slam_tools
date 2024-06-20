#!/bin/bash
ns-process-data images --data ./images/ --output-dir ./ns/ --skip-colmap --colmap-model-path ./sfm/ --skip-image-processing
ns-train splatfacto --data ./ns/


ns-process-data images --data ./images/ --output-dir ./ns/ --camera-type simple_pinhole --feature-type superpoint --matcher-type superglue --no-use-single-camera-mode
