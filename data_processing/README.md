# Data Processing

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

### Process 2D-3D-Semantics Panoramas
```python
(sfss_mmsi) <sfss_mmsi_repo_path>$ python -m data_processing.gen_sid -i "<raw_dataset_path>/2D-3D-Semantics" -o "<sfss_mmsi_path>/datasets/2D-3D-Semantics-1K"
```