# Data Processing

### Process 2D-3D-Semantics Panoramas

```
<raw_dataset_path>/2D-3D-Semantics
    /area_1
        /pano    # equirectangular projections
            ...
    /area_2
    /area_3
    /area_4
    /area_5a
    /area_5b
    /area_6
```

```python
(sfss_mmsi) <sfss_mmsi_repo_path>$ python -m data_processing.gen_sid -i "<raw_dataset_path>/2D-3D-Semantics" -o "<sfss_mmsi_path>/datasets/2D-3D-Semantics-1K"
```

### Process Matterport3D Panoramas

```
<raw_dataset_path>/Matterport3D
    /v1
        /scans
            /D7G3Y4RVNrH
                ...
            /1LXtFkjw3qL
            ...
            /zsNo4HB9uLZ
```

```python
# build
(sfss_mmsi) <matterport_utils_repo_path>/mpview$ make
# unzip
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.unzip_matterport --m3d_path="<raw_dataset_path>/Matterport3D/v1/scans" --scan_id D7G3Y4RVNrH
# process normals
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.process_normals --m3d_path="<raw_dataset_path>/Matterport3D/v1/scans" --scan_id D7G3Y4RVNrH

# generate segmentation maps
(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/D7G3Y4RVNrH/D7G3Y4RVNrH$ mkdir my_segmentation_maps_instances my_segmentation_maps_classes
(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/D7G3Y4RVNrH/D7G3Y4RVNrH$ <matterport_utils_repo_path>/mpview/bin/x86_64/mpview -input_house house_segmentations/D7G3Y4RVNrH.house -input_mesh house_segmentations/D7G3Y4RVNrH.ply -input_segments house_segmentations/D7G3Y4RVNrH.fsegs.json -input_objects house_segmentations/D7G3Y4RVNrH.semseg.json -window 1280 1024 -output_image my_segmentation_maps -seg_maps -v

(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/D7G3Y4RVNrH/D7G3Y4RVNrH$ mv my_segmentation_maps_classes segmentation_maps_classes_pretty
(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/D7G3Y4RVNrH/D7G3Y4RVNrH$ mv my_segmentation_maps_instances segmentation_maps_instances_pretty

# generate panoramic images
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.prepare_matterport --m3d_path="<raw_dataset_path>/Matterport3D/v1/scans" --scan_id D7G3Y4RVNrH --out_path="<sfss_mmsi_path>/datasets/Matterport3D-1K" --out_width=1024 --out_height=512 --warp_depth=True --unpack --types skybox color depth classes instances normal

# process segmentation maps
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.process_semantics --m3d_path="<sfss_mmsi_path>/datasets/Matterport3D-1K" --scan_id D7G3Y4RVNrH
```