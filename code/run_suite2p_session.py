"""Run the kd suite2p pipeline on raw Bergamo sessions attached in /data.

CO-runnable port of kd_suite2p_use_old_rois3.py. For each raw asset
(single-plane-ophys_<subject>_<date>_<time>/pophys/*.tif):

  1. The first epoch in EPOCH_MAP is registered and segmented with suite2p.
  2. Every later epoch is registered to the first epoch's refImg and its
     traces are extracted with the first epoch's ROIs (stat/iscell), so all
     epochs share one ROI set.
  3. siHeader (+ siBase / savefolders) is saved into each suite2p folder.
  4. ddc.main() builds the data dict; suite2p_photostim_single[N]/ become
     data['photostim'] / data['photostim2'] and are written as
     data_photostim*<subject>_<date>.h5.

Environment variables:
    EPOCH_MAP   comma-separated tiff_stem:save_folder pairs; the first entry
                is the ROI-detection epoch. Default:
                spont:spont_pre,spont_slm:photostim_single,
                spontpost_slm:photostim_single2,spontpost:spont_post
    RAW_ASSETS  optional comma-separated raw asset names to process; default
                is every attached single-plane-ophys_* raw asset.

Outputs, per session, in /results/<subject>_<date>/:
    data_photostim<subject>_<date>.h5, data_photostim2<subject>_<date>.h5
    data_main_<subject>_<date>_BCI.h5   (spont traces etc., if any)
    photostim_compact.h5, photostim2_compact.h5   (stimDist, favg_raw, seq)
    suite2p_<folder>/plane0/{F,Fneu,stat,iscell,ops,siHeader}.npy
"""

import copy
import os
import re
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/code")

import suite2p
from suite2p import default_ops as s2p_default_ops
from suite2p.extraction.extract import extract_traces_from_masks
from suite2p.extraction.masks import create_masks

import extract_scanimage_metadata
import data_dict_create_module_bruker as ddc


DATA = Path("/data")
SCRATCH = Path("/scratch")
RESULTS = Path("/results")

DEFAULT_EPOCH_MAP = ("spont:spont_pre,spont_slm:photostim_single,"
                     "spontpost_slm:photostim_single2,spontpost:spont_post")
EPOCH_MAP = [tuple(p.split(":")) for p in
             os.environ.get("EPOCH_MAP", DEFAULT_EPOCH_MAP).split(",") if p.strip()]
RAW_ASSETS = [s.strip() for s in os.environ.get("RAW_ASSETS", "").split(",") if s.strip()]


def kd_ops(pophys, tiff_list, save_path0, save_folder, siHeader):
    """suite2p ops as set in kd_suite2p_use_old_rois3.py."""
    ops = s2p_default_ops()
    ops["data_path"] = [str(pophys)]
    ops["tiff_list"] = tiff_list
    ops["save_path0"] = str(save_path0)
    ops["fast_disk"] = str(save_path0)
    ops["save_folder"] = save_folder
    ops["do_registration"] = True
    ops["save_mat"] = True
    ops["do_bidiphase"] = True
    ops["reg_tif"] = False
    ops["delete_bin"] = False
    ops["keep_movie_raw"] = 0
    ops["fs"] = 20
    ops["nchannels"] = len(np.fromstring(
        siHeader["metadata"]["hChannels"]["channelSave"].strip("[]"), sep=" ", dtype=int))
    ops["tau"] = 1
    ops["nimg_init"] = 500
    ops["nonrigid"] = False
    ops["smooth_sigma"] = .5
    ops["threshold_scaling"] = .7
    ops["batch_size"] = 250
    ops["do_regmetrics"] = False
    ops["allow_overlap"] = False
    return ops


def tiffs_by_stem(pophys):
    stems = defaultdict(list)
    for f in pophys.iterdir():
        if f.suffix == ".tif":
            stems[f.name.rsplit("_", 1)[0]].append(f.name)
    return {k: sorted(v) for k, v in stems.items()}


def process_session(raw):
    m = re.match(r"single-plane-ophys_(\d+)_(\d{4}-\d{2}-\d{2})_", raw.name)
    subject, date = m.group(1), m.group(2)
    tag = f"{subject}_{date}"
    pophys = raw / "pophys"
    work = SCRATCH / tag
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    stems = tiffs_by_stem(pophys)
    print(f"\n=== {raw.name} ===")
    for k, v in sorted(stems.items()):
        print(f"  {k}: {len(v)} TIFFs")

    ref_stem = EPOCH_MAP[0][0]
    if ref_stem not in stems:
        raise RuntimeError(f"ROI-detection epoch {ref_stem!r} not found in {pophys}")

    savefolders = {i: f for i, (_, f) in enumerate(EPOCH_MAP)}
    siBase = {i: (EPOCH_MAP[i][0] if i < len(EPOCH_MAP) else "") for i in range(3)}

    ops_ref = stat_ref = iscell_ref = None
    for ei, (stem, folder) in enumerate(EPOCH_MAP):
        if stem not in stems:
            print(f"NOTE: epoch {stem!r} not present, skipping suite2p_{folder}")
            continue
        tiff_list = stems[stem]
        save_folder = f"suite2p_{folder}"
        siHeader = extract_scanimage_metadata.extract_scanimage_metadata(
            str(pophys / tiff_list[0]))
        ops = kd_ops(pophys, tiff_list, work, save_folder, siHeader)
        if ops_ref is not None:
            ops["roidetect"] = False
            ops["refImg"] = ops_ref["refImg"]
            ops["force_refImg"] = True
        else:
            ops["roidetect"] = True
        print(f"\n--- {stem} -> {save_folder} ({len(tiff_list)} TIFFs, "
              f"roidetect={ops['roidetect']}) ---")
        ops = suite2p.run_s2p(ops)

        plane0 = work / save_folder / "plane0"
        if ops_ref is None:
            ops_ref = ops
            stat_ref = np.load(plane0 / "stat.npy", allow_pickle=True)
            iscell_ref = np.load(plane0 / "iscell.npy", allow_pickle=True)
            print(f"  {len(stat_ref)} ROIs ({int(iscell_ref[:, 0].sum())} cells)")
        else:
            stat_new = copy.deepcopy(stat_ref)
            cell_masks, neuropil_masks = create_masks(stat_new, ops["Ly"], ops["Lx"], ops)
            F, Fneu, _, _ = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
            np.save(plane0 / "F.npy", F)
            np.save(plane0 / "Fneu.npy", Fneu)
            np.save(plane0 / "stat.npy", stat_new)
            np.save(plane0 / "iscell.npy", iscell_ref)
            print(f"  extracted {F.shape} with reference ROIs")

        siHeader["siBase"] = siBase
        siHeader["savefolders"] = savefolders
        np.save(plane0 / "siHeader.npy", siHeader)

        # Registered binaries are only needed for this epoch's extraction.
        for b in plane0.glob("data*.bin"):
            b.unlink()

    data = ddc.main(str(work) + "/", mouse=subject, session=date)
    print(f"ddc.main keys: {sorted(data.keys())}")

    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)
    for h5 in work.glob("data_*.h5"):
        shutil.copy(h5, out / h5.name)
        print(f"Wrote {out / h5.name}")
    # Compact per-epoch files with just what connectivity analyses need
    # (same root-level keys ddct.load_hdf5 reads from data_photostim*.h5).
    for pkey in ("photostim", "photostim2"):
        if pkey not in data:
            continue
        compact = out / f"{pkey}_compact.h5"
        with h5py.File(compact, "w") as f:
            for key in ("stimDist", "favg_raw", "seq"):
                if key in data[pkey]:
                    f.create_dataset(key, data=np.asarray(data[pkey][key]))
        print(f"Wrote {compact}")
    for s2p in work.glob("suite2p_*"):
        dst = out / s2p.name / "plane0"
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ["F.npy", "Fneu.npy", "stat.npy", "iscell.npy", "ops.npy", "siHeader.npy"]:
            src = s2p / "plane0" / fname
            if src.exists():
                shutil.copy(src, dst / fname)
    shutil.rmtree(work, ignore_errors=True)


def main():
    raws = sorted(p for p in DATA.iterdir()
                  if p.is_dir() and p.name.startswith("single-plane-ophys_")
                  and "_processed_" not in p.name and (p / "pophys").is_dir())
    if RAW_ASSETS:
        raws = [p for p in raws if p.name in RAW_ASSETS]
    if not raws:
        print(f"ERROR: no raw assets found in {DATA}: {[p.name for p in DATA.iterdir()]}",
              file=sys.stderr)
        sys.exit(3)
    print(f"EPOCH_MAP = {EPOCH_MAP}")
    print(f"Sessions: {[p.name for p in raws]}")

    failed = []
    for raw in raws:
        try:
            process_session(raw)
        except Exception:
            traceback.print_exc()
            failed.append(raw.name)
    print(f"\nDone. Failed: {len(failed)} / {len(raws)} {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
