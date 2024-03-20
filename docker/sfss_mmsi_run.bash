#!/bin/bash

docker run -it \
   --name sfss_mmsi_cu117 \
   --gpus all \
   --mount "type=bind,src=/home/guttikonda/Documents/OriginalWorks/Personal/SFSS-MMSI,dst=/home/user/sfss_mmsi" \
   --workdir /home/user \
   --ipc=host \
   sfss_mmsi_cu117 \
   bash
