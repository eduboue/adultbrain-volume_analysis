"""
Lightweight wrapper around segmented adult brain volumes with default region
lookups and simple voxel/volume utilities. 
Based on AZBA (Adult Zebrafish Brain Atlas) (Kenney et al. 2021 eLife, doi: 10.7554/eLife.69988).

Author: Erik Duboué <eduboue@gmail.com>
Updated: Jan 2, 2026
Happy New Year!
"""

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Dict, Tuple, Optional, Iterable

class AdultBrain:
    """
    Segmented adult brain volume with per-region voxel/volume statistics
    and **built-in default region lookup** (can be overridden).
    """

    # Regions Names Lookup Table (LUT)
    DEFAULT_REGION_LUT: Dict[int, str] = {
        0:  "Clear Label",
        1:  "A (anterior thalamic nucleus)",
        2:  "AC (anterior cerebellar tract)",
        3:  "ALLN (anterior lateral line nerves)",
        4:  "AON (anterior octaval nucleus)",
        5:  "APN (accessory pretectal nucleus)",
        6:  "ATN (anterior tuberal nucleus)",
        7:  "BSTa (bed nucleus of the stria terminalis, anterior division)",
        8:  "BSTm (bed nucleus of the stria terminalis, medial division)",
        9:  "BSTpd (bed nucleus of the stria terminalis, posterior division)",
        10: "C (central canal)",
        11: "Cans (ansulate commissure)",
        12: "Cantd (anterior commissure, dorsal part)",
        13: "Cantv (anterior commissure, ventral part)",
        14: "CC (cerebellar crest)",
        15: "CCe-g (cerebellar corpus, granular layer)",
        16: "CCe-m (cerebellar corpus, molecular layer)",
        17: "Ccer (cerebellar commissure)",
        18: "Cgus (commissure of the secondary gustatory nuclei)",
        19: "Chab (habenular commissure)",
        20: "Chor (horizontal commissure)",
        21: "CIL (central nucleus of the inferior lobe)",
        22: "Cinf (commissura infima of Haller)",
        23: "CM (mammillary body)",
        24: "CO/OT (optic chiasm / optic tract)",
        25: "CON (caudal octavolateralis nucleus)",
        26: "CP (central posterior thalamic nucleus)",
        27: "CPN (central pretectal nucleus)",
        28: "CPop (supraoptic commissure)",
        29: "Cpost (posterior commissure)",
        30: "Ctec (tectal commissure)",
        31: "Ctub (commissure of the posterior tuberculum)",
        32: "Cven (ventral rhombencephalic commisure)",
        33: "DAO (dorsal accessory optic nucleus)",
        34: "Dc (central zone of dorsal telencephalon area)",
        35: "DH (dorsal horn)",
        36: "DIL (diffuse nucleus of the inferior lobe)",
        37: "DiV (diencephalic ventricle)",
        38: "DIV (trochlear decussation)",
        39: "Dl (lateral zone of the dorsal telencephalon)",
        40: "Dm (medial zone of dorsal telencephalon)",
        41: "DON (descending octaval nucleus)",
        42: "DOT (dorsomedial optic tract)",
        43: "Dp (posterior zone of dorsal telencephalon area)",
        44: "DP (dorsal posterior thalamic nucleus)",
        45: "DR (dorsal root)",
        46: "DTN (dorsal tegmental nucleus)",
        47: "DV (descending trigeminal root)",
        48: "E (epiphysis)",
        49: "ECL (external cellular layer of olfactory bulb)",
        50: "EG (granular eminence)",
        51: "EmTl (lateral thalamic eminence)",
        52: "EmTm (medial thalamic eminence)",
        53: "EmTr (rostral thalamic eminence)",
        54: "ENd (entopeduncular nucleus, dorsal part)",
        55: "ENv (entopeduncular nucleus, ventral part)",
        56: "EW (Edinger-Westphal nucleus)",
        57: "Flv (ventral part of lateral funiculus)",
        58: "FR (habenulo-interpeduncular tract)",
        59: "Fv (ventral funiculus)",
        60: "GC (central gray)",
        61: "GL (glomerular layer of olfactory bulb)",
        62: "Had (dorsal habenular nucleus)",
        63: "Hav (ventral habenular nucleus)",
        64: "Hc (caudal zone of periventricular hypothalamus)",
        65: "Hd (dorsal zone of periventricular hypothalamus)",
        66: "Hv (ventral zone of periventricular hypothalamus)",
        67: "I (Intermediate thalamic nucleus)",
        68: "IAF (inner arcuate fibers)",
        69: "ICL (internal cellular layer of olfactory bulb)",
        70: "IMRF (intermediate reticular formation)",
        71: "IN (Intermediate nucleus)",
        72: "IO (inferior olive)",
        73: "IR (inferior raphe)",
        74: "IRF (inferior reticular formation)",
        75: "LC (locus coeruleus)",
        76: "LCa (caudal lobe of cerebellum)",
        77: "LFB (lateral forebrain bundle)",
        78: "LH (lateral hypothalamic nucleus)",
        79: "LLF (lateral longitudinal fascicle)",
        80: "LOT (lateral olfactory tract)",
        81: "LR (lateral recess of diencephalic ventricle)",
        82: "LRN (lateral reticular nucleus)",
        83: "MAC (Mauthner cell)",
        84: "MaON (magnocellular octaval nucleus)",
        85: "MFB (medial forebrain bundle)",
        86: "MFN (medial funicular nucleus)",
        87: "MLF (medial longitudinal fascicle)",
        88: "MON (medial octavolateralis nucleus)",
        89: "MOT (medial olfactory tract)",
        90: "NC (commissural nucleus of Cajal)",
        91: "NDV (nucleus of the descending trigeminal root)",
        92: "NI (isthmic nucleus)",
        93: "NIn (interpeduncular nucleus)",
        94: "NLL (nucleus of the lateral lemniscus)",
        95: "nLOT-a (nucleus of the lateral olfactory tract, anterior part)",
        96: "nLOT-i (nucleus of the lateral olfactory tract, intermediate part)",
        97: "nLOT-p (nucleus of the lateral olfactory tract, posterior part)",
        98: "NLV (nucleus lateralis valvulae)",
        99: "NMLF (nucleus of the medial longitudinal fascicle)",
        100: "NR (red nucleus)",
        101: "OENc (octavolateralis efferent neurons, caudal part)",
        102: "OENr (octavolateralis efferent neurons, rostral part)",
        103: "P (posterior thalamic nucleus)",
        104: "PC (posterior cerebellar tract)",
        105: "PCN (paracommissural nucleus)",
        106: "PGa (anterior preglomerular nucleus)",
        107: "PGc (caudal preglomerular nucleus)",
        108: "PGl (lateral preglomerular nucleus)",
        109: "PGm (medial preglomerular nucleus)",
        110: "PGZ (periventricular gray zone of optic tectum)",
        111: "PL (perilemniscal nucleus)",
        112: "PLLN (posterior lateral line nerve)",
        113: "PM (magnocellular preoptic nucleus)",
        114: "PMg (gigantocellular part of magnocellular preoptic nucleus)",
        115: "PO (posterior pretectal nucleus)",
        116: "POF (primary olfactory fiber layer)",
        117: "PON (posterior octaval nucleus)",
        118: "PPa (parvocellular preoptic nucleus, anterior part)",
        119: "PPd (periventricular pretectal nucleus, dorsal part)",
        120: "PPp (parvocellular preoptic nucleus, posterior part)",
        121: "PPv (periventricular pretectal nucleus, ventral part)",
        122: "PR (posterior recess of diencephalic ventricle)",
        123: "PSm (magnocellular superficial pretectal nucleus)",
        124: "PSp (parvocellular superficial pretectal nucleus)",
        125: "PTN (posterior tuberal nucleus)",
        126: "PVO (paraventricular organ)",
        127: "R (rostrolateral nucleus)",
        128: "RT (rostral tegmental nucleus)",
        129: "RV (rhombencephalic ventricle)",
        130: "SC (suprachiasmatic nucleus)",
        131: "SCO (subcommissural organ)",
        132: "SD (dorsal sac)",
        133: "SG (subglomerular nucleus)",
        134: "SGN (secondary gustatory nucleus)",
        135: "SGT (secondary gustatory tract)",
        136: "SO (secondary octaval population)",
        137: "SR (superior raphe)",
        138: "SRF (superior reticular formation)",
        139: "SRN (superior reticular nucleus)",
        140: "T (tangential nucleus)",
        141: "TBS (bulbo-spinal tract)",
        142: "TelV (telencephalic ventricles)",
        143: "TeO (optic tectum)",
        144: "TeV (tectal ventricle)",
        145: "TGN (tertiary gustatory nucleus)",
        146: "TL (longitudinal torus)",
        147: "TLa (lateral torus)",
        148: "TMCa (anterior mesencephalo-cerebellar tract)",
        149: "TMCp (posterior mesencephalo-cerebellar tract)",
        150: "TPM (pretecto-mammillary tract)",
        151: "TPp (periventricular nucleus of posterior tuberculum)",
        152: "TSc (central nucleus of semicircular torus)",
        153: "TSvl (ventrolateral nucleus of semicircular torus)",
        154: "TTB (tecto-bulbar tract)",
        155: "TTBc (crossed tecto-bulbar tract)",
        156: "TTBr (uncrossed tecto-bulbar tract)",
        157: "TVS (vestibulo-spinal tract)",
        158: "Val-g (lateral division of valvula cerebelli, granular layer)",
        159: "Val-m (lateral division of valvula cerebelli, molecular layer)",
        160: "Vam-g (medial division of valvula cerebelli, granular layer)",
        161: "Vam-m (medial division of valvula cerebelli, molecular layer)",
        162: "VAO (ventral accessory optic nucleus)",
        163: "Vas (vascular lacuna of area postrema)",
        164: "Vc (central nucleus of ventral telencephalon area)",
        165: "Vd-dd (dorsal zone of ventral telencephalon)",
        166: "Vd-vd (ventral zone of ventral telencephalon)",
        167: "Vdd (dorsal most zone of ventral telencephalon)",
        168: "Vl (lateral nucleus of ventral telencephalon area)",
        169: "VL (ventrolateral thalamic nucleus)",
        170: "VM (ventromedial thalamic nucleus)",
        171: "VOT (ventrolateral optic tract)",
        172: "Vp (postcommissural nucleus of ventral telencephalon area)",
        173: "Vs (supracommissural nucleus of ventral telencephalon area)",
        174: "Vv (ventral nucleus of ventral telencephalon area)",
        175: "ZL (zona limitans)",
        176: "III (oculomotor nerve)",
        177: "IIIm (oculomotor nucleus)",
        178: "IV (trochlear nerve)",
        179: "IVm (trochlear nucleus)",
        180: "V (trigeminal nerve)",
        181: "Vmd (trigeminal motor nucleus, dorsal part)",
        182: "Vmn (mesencephalic nucleus of the trigeminal nerve)",
        183: "Vmv (trigeminal motor nucleus, ventral part)",
        184: "Vmvr (ventral trigeminal motor root)",
        185: "Vsm (primary sensory trigeminal nucleus)",
        186: "Vsr (sensory root of the trigeminal nerve)",
        187: "VImc (caudal abducens nerve motor nucleus)",
        188: "VImr (rostral abducens nerve motor nucleus)",
        189: "VIILo (facial lobe)",
        190: "VIIm (facial motor nucleus)",
        191: "VIImr (facial motor root)",
        192: "VIIs (sensory root of the facial nerve)",
        193: "VIII (octaval nerve)",
        194: "IXLo (glossopharyngeal lobe)",
        195: "IXm (glossopharyngeal nerve motor nucleus)",
        196: "X (vagal nerve)",
        197: "XLo (vagal lobe)",
        198: "Xm (vagal motor nucleus)",
        900: "UnkD (unknown diencephalon)",
        901: "UnkMS (unknown mesencephalon)",
        902: "UnkR (unkonwn rhombencephalon)",
        903: "UnkSC (unknown spinal cord)",
        904: "UnkVT (unknown ventral telencephalon)",
    }

    def __init__(self, stack: NDArray[np.float64], region_lookup: Optional[Dict[int, str]] = None):
        if stack.ndim != 3:
            raise ValueError(f"Expected a 3D array, got shape {stack.shape}")
        self.stack: NDArray[np.float64] = stack
        self._volumes_df: Optional[pd.DataFrame] = None
        self._volumes_series: Optional[pd.Series] = None
        self._voxel_conversion: Optional[Tuple[float, float, float]] = None
        # Start with default LUT; allow override via constructor
        self._region_lookup: Dict[int, str] = dict(self.DEFAULT_REGION_LUT)
        if region_lookup:
            self.set_region_lookup(region_lookup)

    def __str__(self) -> str:
        lines = []
        lines.append(f"Stack is a {self.stack.ndim}D array with shape {self.stack.shape}")
        lines.append(
            "Voxel conversion not set" if self.voxel_conversion is None
            else f"Voxel conversion set to: {self._voxel_conversion!r}"
        )
        lines.append(f"Region lookup has {len(self._region_lookup)} entries (default+custom)")
        lines.append("Volumes not yet computed" if self._volumes_df is None else "Volumes computed")
        return "\n".join(lines)

    # Properties - Needed to set/change class params
    @property
    def volumes(self) -> Optional[pd.DataFrame]:
        return self._volumes_df

    @property
    def volumes_series(self) -> Optional[pd.Series]:
        return self._volumes_series

    @property
    def dimensions(self) -> Tuple[int, ...]:
        return tuple(self.stack.shape)

    @property
    def region_labels(self) -> Iterable[int]:
        return np.unique(self.stack.astype(np.int64, copy=False)).tolist()

    @property
    def voxel_conversion(self) -> Optional[Tuple[float, float, float]]:
        return self._voxel_conversion

    @voxel_conversion.setter
    def voxel_conversion(self, ratios: Tuple[float, float, float]) -> None:
        if len(ratios) != 3:
            raise ValueError("voxel_conversion must be a 3-tuple like (sx, sy, sz).")
        self._voxel_conversion = tuple(float(r) for r in ratios)
        print("Set voxel conversion statistics to:", self._voxel_conversion)

    # helper methods
    def set_region_lookup(self, lut: Dict[int, str], merge: bool = True) -> None:
        """
        Provide/extend a {region_id: 'Region Name'} mapping.
        If merge=True (default), updates defaults with provided entries.
        If merge=False, replaces the current LUT entirely.
        """
        cleaned = {int(k): str(v) for k, v in lut.items()}
        if merge:
            self._region_lookup.update(cleaned)
        else:
            self._region_lookup = cleaned

    def load_region_lookup_from_itksnap(self, filepath: str, keep_zero: bool = True, merge: bool = True) -> None:
        """
        Load LUT from an ITK-SNAP label file and merge/replace.
        """
        lut: Dict[int, str] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if '"' in s:
                    parts = s.split('"')
                    left = parts[0].strip()
                    label = parts[1].strip()
                    try:
                        idx = int(left.split()[0])
                    except Exception:
                        continue
                    if idx == 0 and not keep_zero:
                        continue
                    lut[idx] = label
        self.set_region_lookup(lut, merge=merge)
        print(f"Loaded {len(lut)} labels from: {filepath} (merge={merge})")

    # class method - constuctur
    @classmethod
    def from_file(cls, filepath: str, region_lookup: Optional[Dict[int, str]] = None) -> "AdultBrain":
        stack: NDArray[np.float64] = nib.load(filepath).get_fdata()
        return cls(stack, region_lookup=region_lookup)

    # math
    def compute_volumes(
        self,
        include_zero: bool = False,
        as_dataframe: bool = True,
        sort_by: str = "region_id",   # 'region_id' or 'measure'
        ascending: bool = True,
    ):
        labels, counts = np.unique(self.stack.astype(np.int64, copy=False), return_counts=True)
        if not include_zero:
            mask = labels != 0
            labels, counts = labels[mask], counts[mask]

        if self.voxel_conversion is not None:
            voxel_volume = float(np.prod(self.voxel_conversion))
            measures = counts.astype(np.float64) * voxel_volume
            measure_col = f"Volume (voxel_size={self._voxel_conversion})"
        else:
            measures = counts.astype(np.int64)
            measure_col = "Voxels"

        df = pd.DataFrame({
            "region_id": labels,
            "region_label": [self._region_lookup.get(int(r), f"Region {int(r)}") for r in labels],
            "measure": measures,
        })

        if sort_by not in {"region_id", "measure"}:
            raise ValueError("sort_by must be 'region_id' or 'measure'")
        df = df.sort_values(sort_by, ascending=ascending, kind="mergesort").reset_index(drop=True)
        df = df.rename(columns={"measure": measure_col})

        self._volumes_df = df
        self._volumes_series = pd.Series(
            measures, index=[f"Region {int(i)}" for i in labels], name=measure_col
        )
        return df if as_dataframe else self._volumes_series

    # write to file
    def write_regions(self, output_file_path: str) -> None:
        if self._volumes_df is None:
            raise RuntimeError("No region volumes computed yet. Call compute_volumes() first.")
        self._volumes_df.to_csv(output_file_path, index=False)
        print("Computed volumes saved to:", output_file_path)
