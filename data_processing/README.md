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
                house_segmentations.zip
                matterport_skybox_images.zip
                undistorted_camera_parameters.zip
                undistorted_color_images.zip
                undistorted_depth_images.zip
                undistorted_normal_images.zip
            /1LXtFkjw3qL
            ...
            /zsNo4HB9uLZ
```

```python
# build
(sfss_mmsi) <matterport_utils_repo_path>/mpview$ make
```

for each scan:
```python
# unzip
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.unzip_matterport --m3d_path="<raw_dataset_path>/Matterport3D/v1/scans" --scan_id <scan_id>
# process normals
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.process_normals --m3d_path="<raw_dataset_path>/Matterport3D/v1/scans" --scan_id <scan_id>

# generate segmentation maps
(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/<scan_id>/<scan_id>$ mkdir my_segmentation_maps_instances my_segmentation_maps_classes
(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/<scan_id>/<scan_id>$ <matterport_utils_repo_path>/mpview/bin/x86_64/mpview -input_house house_segmentations/<scan_id>.house -input_mesh house_segmentations/<scan_id>.ply -input_segments house_segmentations/<scan_id>.fsegs.json -input_objects house_segmentations/<scan_id>.semseg.json -window 1280 1024 -output_image my_segmentation_maps -seg_maps -v

(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/<scan_id>/<scan_id>$ mv my_segmentation_maps_classes segmentation_maps_classes_pretty
(sfss_mmsi) <raw_dataset_path>/Matterport3D/v1/scans/<scan_id>/<scan_id>$ mv my_segmentation_maps_instances segmentation_maps_instances_pretty

# generate panoramic images
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.prepare_matterport --m3d_path="<raw_dataset_path>/Matterport3D/v1/scans" --scan_id <scan_id> --out_path="<sfss_mmsi_path>/datasets/Matterport3D-1K" --out_width=1024 --out_height=512 --warp_depth=True --unpack --types skybox color depth classes instances normal

# process segmentation maps
(sfss_mmsi) <matterport_utils_repo_path>$ python -m preparepano.process_semantics --m3d_path="<sfss_mmsi_path>/datasets/Matterport3D-1K" --scan_id <scan_id>
```

### Process Ricoh3D Panoramas

```
<raw_dataset_path>/Ricoh3D
    /parking1
        /pano    # equirectangular projections
            ...
```

```python
(sfss_mmsi) <sfss_mmsi_repo_path>$ python -m data_processing.gen_ricoh -i "<raw_dataset_path>/Ricoh3D" -o "<sfss_mmsi_path>/datasets/Ricoh3D-1K"
```