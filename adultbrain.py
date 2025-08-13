import nibabel as nib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm

class AdultBrain:
    """
    The AdultBrain class represents a segmented adult brain volume presumed to have 
    been imaged from Light Sheet methods. The class provides methods to load, analyze, 
    and export per-region voxel or volume statistics.

    This class is designed for neuroimaging datasets stored in `.nii.gz` format
    and supports computation of per-region voxel counts, with optional
    conversion to physical units based on voxel dimensions.

    Attributes
    ----------
    stack : NDArray[np.float64]
        3D NumPy array containing segmentation labels. This is the main image file,
        where (z,x,y) represents the number of slices (z) and the x,y dimensions (x,y)
        per slice.
        
    _volumes : pd.Series | None
        Cached Series containing per-region voxel or volume statistics.
        This Series object will be filled in upon execution of 
        compute_volumes() method. Otherwise set to None.
        
    _voxel_conversion : tuple[float, float, float] | None
        Voxel size along (x, y, z) in physical units (e.g., millimeters).
    """
    def __init__(self, stack: NDArray[np.float64]):
        """
        Initialize an AdultBrain object.

        Parameters
        ----------
        stack : NDArray[np.float64]
            A 3D NumPy array representing the brain segmentation volume.
            Each voxel contains a numeric region label.

        Raises
        ------
        ValueError
            If the input array is not 3-dimensional.
        """
        if stack.ndim != 3:
            raise ValueError(f"Expected a 3D array, got shape {stack.shape}")
        self.stack: NDArray[np.float64] = stack
        self._volumes: pd.Series | None = None 
        self._voxel_conversion: tuple[float, float, float] | None = None
    
    @property
    def dimensions(self) -> tuple[int, ...]:
        """
        Get the dimensions (shape) of the segmentation volume.

        Returns
        -------
        tuple[int, ...]
            Shape of the stack as (m, n, p).
        """
        return np.shape(self.stack)
    
    @property
    def region_labels(self) -> tuple[int, ...]:
        """
        Get the unique region labels present in the segmentation.

        Returns
        -------
        set
            Unique region IDs in the dataset.
        """
        return set(self.stack.flatten().tolist())
    
    @property
    def voxel_conversion(self) -> tuple[float, float, float]:
        """
        Get the voxel conversion tuple.

        Returns
        -------
        tuple[float, float, float]
            Voxel size along (x, y, z), typically in mm.
        """
        return self._voxel_conversion

    @voxel_conversion.setter
    def voxel_conversion(self, ratios: tuple[float, float, float]) -> None:
        """
        Set the voxel conversion tuple.

        Parameters
        ----------
        ratios : tuple[float, float, float]
            Voxel size along (x, y, z), typically in mm.

        Raises
        ------
        ValueError
            If ratios is not a length-3 tuple.
        """
        if len(ratios) != 3:
            raise ValueError("voxel_conversion must be a 3-tuple like (sx, sy, sz).")
        self._voxel_conversion = tuple(float(r) for r in ratios)
        print("Set voxel conversion statistics to: ", ratios)

    @classmethod
    def from_file(cls, filepath: str) -> "AdultBrain":
        """
        Create an AdultBrain object from a `.nii.gz` neuroimaging file.

        Parameters
        ----------
        filepath : str
            Path to the `.nii.gz` file.

        Returns
        -------
        AdultBrain
            An instance containing the loaded segmentation volume.
        """
        stack: NDArray[np.float64] = nib.load(filepath).get_fdata()
        return cls(stack)

    def compute_volumes(self):
        """
        Compute per-region voxel counts or volumes.

        Uses the segmentation labels in `self.stack` to count the number of voxels
        for each region. If `voxel_conversion` is set, converts voxel counts to
        physical volume units.

        Returns
        -------
        pd.Series
            Indexed by region name (e.g., 'Region 1'), containing voxel or volume
            measurements.

        Notes
        -----
        The results are cached in `self._volumes` for later use.
        """
        if self.voxel_conversion is None:
            print("No Voxel Conversion given. Performing voxel measures in pixels.")
        region_labels: set = self.region_labels
        dimensions: tuple[int, ...] = self.dimensions
        
        index: list = [f"Region {int(i)}" for i in region_labels]
        SO: pd.Series = pd.Series(np.zeros(len(index)),index=index,name="Voxel Measures")

        stack_container: NDArray[np.float64] = np.zeros(dimensions[0])
        for region in tqdm(region_labels, desc="Regions", unit="region",leave=False):
            for slice_num, slice_2d in enumerate(self.stack):
                stack_container[slice_num] = np.sum(slice_2d.flatten() == region)
            SO[f"Region {np.int16(region)}"] = stack_container.sum()
        
        self._volumes = SO    
        return SO
    
    def write_regions(self, output_file_path: str) -> None:
        """
        Save the computed per-region volumes to a CSV file.

        Parameters
        ----------
        output_file_path : str
            Path to the output CSV file.

        Raises
        ------
        RuntimeError
            If `compute_volumes()` has not been called yet.
        """
        if self._volumes is None:
            raise RuntimeError("No region volumes computed yet. Call compute_volumes() first and then save to file.")
        self._volumes.to_csv(output_file_path)
        print("Computed volumes save to: ", output_file_path)
        