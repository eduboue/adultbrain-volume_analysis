from adultbrain import AdultBrain
import pandas as pd
import numpy as np

if __name__ == "__main__":
    file_path: str = "/Users/erikduboue/Downloads/AZBA_segmentation_in_fish_space.nii.gz"
    s: "AdultBrain" = AdultBrain.from_file(file_path)
    print("Dimensions are: ", s.dimensions)
    vc: tuple[float, float, float] = (4.2, 1.06, 1.06)
    s.voxel_conversion = vc
    ser: pd.Series = s.compute_volumes()
    s.write_regions("/Users/erikduboue/Downloads/AZBA_segmentation_in_fish_space.csv")