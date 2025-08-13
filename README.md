# AdultBrain Volume Analysis

This repository provides a Python class, `AdultBrain`, for loading 3D brain segmentation data in NIfTI (.nii.gz) format, computing region volumes, and saving results to CSV. It supports voxel-to-physical size conversions and is designed for neuroscience workflows.

---
**Features**

* Load 3D volumetric data from .nii.gz files using NiBabel.
* Extract dataset dimensions and unique region labels.
* Compute voxel or physical volumes for each labeled brain region.
* Optionally apply voxel size conversions for real-world volume measures.
* Save computed volumes to a CSV file.

---
**Requirements**

Install the required dependencies before running the scripts:

```bash
conda create -n adult_brain_data python=3.12 numpy pandas matplotlib tqdm nibabel
conda activate adult_brain_data
```
---
**Class Overview**

AdultBrain

Purpose:
Encapsulates 3D brain segmentation data and provides methods for analysis.

Initialization:
```python
AdultBrain(stack: np.ndarray)
```

* `stack`: 3D NumPy array of brain segmentation labels.

**Key Properties**:

* `dimensions` → Shape of the dataset (x, y, z).
* `region_labels` → Set of unique region IDs found in the data.
* `voxel_conversion` → Optional (sx, sy, sz) tuple for voxel size in mm.

**Class Methods**:

* `from_file(filepath)` → Load .nii.gz and return an AdultBrain instance.

**Instance Methods**:

* `compute_volumes()` → Compute voxel counts (or physical volumes if voxel_conversion is set) per region and store them internally.
* `write_regions(output_file_path)` → Save computed volumes to a CSV file.

---

**Example Usage**

Below is the content of `main.py` demonstrating how to use the AdultBrain class.

```python
from adultbrain import AdultBrain
import pandas as pd

if __name__ == "__main__":
    # Path to your segmentation file
    file_path = "/path/to/AZBA_segmentation_in_fish_space.nii.gz"

    # Load the brain segmentation
    s = AdultBrain.from_file(file_path)

    # Print dataset information
    print("Dimensions are: ", s.dimensions)
    print("Region Labels are: ", s.region_labels)

    # Set voxel conversion (sx, sy, sz in mm)
    s.voxel_conversion = (4.2, 1.06, 1.06)

    # Compute volumes
    volumes = s.compute_volumes()
    print(volumes.head(10))

    # Save to CSV
    s.write_regions("region_volumes.csv")
```

---
**Workflow**

1. Load the .nii.gz file:

```python
s = AdultBrain.from_file("segmentation.nii.gz")
```

2. Set voxel size (optional):

```python
s.voxel_conversion = (4.2, 1.06, 1.06)
```

3. 	Compute volumes:

```python
volumes = s.compute_volumes()
```

4. Save results:

```python
s.write_regions("region_volumes.csv")
```

---
**Output**

* Printed Info: Dataset shape and available region labels.
* CSV File: Table containing each region and its corresponding voxel or physical volume.

---

**Notes**

* If voxel_conversion is not set, volumes are reported as raw voxel counts.
* Ensure your .nii.gz segmentation file contains integer labels for regions.
* The progress bar from tqdm will show computation progress for large datasets.
