#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author-style SDNet saliency (dataloader_sod) + stitching (first-write-wins) + 2D CA-CFAR
+ detections.csv/mat export and overlay visualization.

What changed vs previous version:
- After computing CFAR det_mask (unchanged), we now:
  - Export detections.csv with columns:
      row, col, range_m, fd_Hz, power_linear, power_dB
  - Export detections.mat with same fields
  - Save cfar_det_overlay.png: red dots overlay on background power map
All CFAR logic and thresholds remain identical.

Usage example:
  python3 run_rd_sdnet_cfar.py \
    --datadir   /mnt/data/wsy/stdnet/stdnet \
    --testdata  patches_128 \
    --savedir   /mnt/data/wsy/stdnet/stdnet/out \
    --evaluate  /mnt/data/wsy/stdnet/stdnet/checkpoints/sdneta_from_pretrained.pth \
    --model sdneta --train-config sdnet-a --inference-config baseline --bn --gpu 0 --size 128 \
    --patch_index /mnt/data/wsy/stdnet/stdnet/patches_128/patch_index.mat \
    --meta        /mnt/data/wsy/stdnet/stdnet/rd_meta.mat \
    --rd_power    /mnt/data/wsy/stdnet/stdnet/rd_power.mat \
    --thr 0.5 --Pfa 1e-6
"""

import os
import argparse
import time
import warnings
from typing import List, Dict, Any, Optional
import csv

import numpy as np
import scipy.io as sio
import skimage.io as skio
from skimage.transform import resize
from PIL import Image

import torch
import torch.nn.functional as F

import models
from utils import state_from_training
from dataloader_sod import prepare_test_data as get_saldata

warnings.filterwarnings("ignore", category=UserWarning)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ============================== MATLAB helpers (robust) ==============================

def _squeeze(x):
    try:
        return np.squeeze(x)
    except Exception:
        return x

def _mat_struct_to_dict(obj):
    if isinstance(obj, np.void) and getattr(obj, "dtype", None) is not None and obj.dtype.names:
        return {k: obj[k] for k in obj.dtype.names}
    if isinstance(obj, np.ndarray) and getattr(obj, "dtype", None) is not None and obj.dtype.names:
        return {k: obj[k] for k in obj.dtype.names}
    return obj

def _to_scalar(x, name=""):
    arr = np.array(x, dtype=object)
    if arr.size == 0:
        raise ValueError(f"Empty scalar for '{name}'.")
    v = arr.ravel()[0]
    if isinstance(v, (list, tuple, np.ndarray)):
        v = np.array(v).ravel()[0]
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f"Cannot convert '{name}' to scalar: {type(x)}") from e

def _flatten_object_array(a):
    out = []
    for e in np.ravel(a):
        if isinstance(e, (list, tuple)):
            out.extend([np.asarray(x) for x in e])
        elif isinstance(e, np.ndarray):
            if e.dtype == object:
                out.extend(_flatten_object_array(e))
            else:
                out.append(e)
        else:
            out.append(np.asarray(e))
    return out

def _to_1d_float_axis(x, name="", expected_len=None):
    if isinstance(x, (list, tuple)):
        vec = np.asarray(x, dtype=np.float64).ravel()
    elif isinstance(x, np.ndarray):
        if x.dtype == object:
            parts = _flatten_object_array(x)
            best = None
            for p in parts:
                try:
                    pp = np.asarray(p, dtype=np.float64).ravel()
                    if best is None or pp.size > best.size:
                        best = pp
                except Exception:
                    continue
            if best is None:
                raise ValueError(f"Axis '{name}' is object array but has no numeric content.")
            vec = best
        else:
            vec = np.asarray(x, dtype=np.float64).ravel()
    else:
        vec = np.asarray(x, dtype=np.float64).ravel()

    if expected_len is not None and vec.size != expected_len:
        if vec.size > expected_len:
            vec = vec[:expected_len]
        else:
            raise ValueError(f"Axis '{name}' length {vec.size} != expected {expected_len}.")
    return vec

def _get_field(rec, names):
    if isinstance(rec, dict):
        for n in names:
            if n in rec:
                return _squeeze(rec[n])
        return None
    if isinstance(rec, np.void) and getattr(rec, "dtype", None) is not None and rec.dtype.names:
        for n in rec.dtype.names:
            if n in names:
                return _squeeze(rec[n])
        return None
    for n in names:
        if hasattr(rec, n):
            return _squeeze(getattr(rec, n))
    return None

def _to_str_name(name):
    if isinstance(name, str):
        return name
    if isinstance(name, np.ndarray):
        if name.dtype.kind in ("U", "S"):
            return "".join(np.ravel(name).tolist())
        try:
            return str(name.item())
        except Exception:
            return str(name)
    return str(name)

# ============================== Loaders ==============================

def load_meta(meta_path):
    d = sio.loadmat(meta_path, squeeze_me=True)
    if "meta" in d:
        m = d["meta"]
        meta = m if isinstance(m, dict) else _mat_struct_to_dict(m)
        if isinstance(meta, dict):
            meta = {k: _squeeze(v) for k, v in meta.items()}
        else:
            raise ValueError("Unrecognized 'meta' structure.")
    else:
        meta = {k: _squeeze(v) for k, v in d.items() if not k.startswith("__")}

    Nd = int(round(_to_scalar(meta.get("Nd", 0), "Nd")))
    Nr = int(round(_to_scalar(meta.get("Nr", 0), "Nr")))
    if Nd <= 0 or Nr <= 0:
        raise ValueError(f"Invalid Nd/Nr from meta: Nd={Nd}, Nr={Nr}")

    r_axis_raw  = meta.get("r_axis", None)
    fd_axis_raw = meta.get("fd_axis", None)
    if r_axis_raw is None or fd_axis_raw is None:
        raise ValueError("meta must contain 'r_axis' and 'fd_axis'.")

    try:
        r_axis  = _to_1d_float_axis(r_axis_raw,  "r_axis",  expected_len=Nr)
    except Exception:
        r_axis  = _to_1d_float_axis(r_axis_raw,  "r_axis",  expected_len=None)
    try:
        fd_axis = _to_1d_float_axis(fd_axis_raw, "fd_axis", expected_len=Nd)
    except Exception:
        fd_axis = _to_1d_float_axis(fd_axis_raw, "fd_axis", expected_len=None)

    if r_axis.size != Nr or fd_axis.size != Nd:
        if r_axis.size == Nd and fd_axis.size == Nr:
            print("[INFO] Detected swapped axes in meta; swapping r_axis <-> fd_axis to match (Nd,Nr).")
            r_axis, fd_axis = fd_axis, r_axis
        else:
            raise ValueError(f"Axis size mismatch: fd({fd_axis.size}) vs Nd({Nd}), r({r_axis.size}) vs Nr({Nr})")

    extras = {}
    for key in ["lambda", "fc", "c"]:
        if key in meta:
            try:
                extras[key] = float(_to_scalar(meta[key], key))
            except Exception:
                pass

    return Nd, Nr, r_axis.astype(np.float64), fd_axis.astype(np.float64), extras

def load_power(rd_power_path, Nd_expected=None, Nr_expected=None):
    d = sio.loadmat(rd_power_path, squeeze_me=True)
    key = None
    for k in ["P", "RD_power", "power", "rd_power"]:
        if k in d:
            key = k
            break
    if key is None:
        arr_keys = [k for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
        if not arr_keys:
            raise ValueError("No array found in rd_power.mat")
        key = arr_keys[0]
    P = np.asarray(d[key]).astype(np.float64)
    if P.ndim != 2:
        raise ValueError("rd_power must be 2D.")
    if Nd_expected is not None and Nr_expected is not None:
        if P.shape != (Nd_expected, Nr_expected):
            raise ValueError(f"P shape {P.shape} != ({Nd_expected}, {Nr_expected})")
    return P

def load_patch_rec(patch_index_path):
    d = sio.loadmat(patch_index_path, squeeze_me=True)
    candidates = []
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, (np.ndarray, np.void, list)):
            candidates.append(k)
    if not candidates:
        raise ValueError("No valid variables in patch_index.mat")

    key = None
    for k in ["rec_list", "recs", "patches", "index", "tiles"]:
        if k in d:
            key = k
            break
    if key is None:
        key = candidates[0]

    container = d[key]
    recs = []

    def _push(rec_like):
        rec = rec_like if isinstance(rec_like, dict) else _mat_struct_to_dict(rec_like)
        name = _get_field(rec, ["id", "name", "filename", "file", "patch_name"])
        r0   = _get_field(rec, ["row0", "r0", "row", "y", "top"])
        c0   = _get_field(rec, ["col0", "c0", "col", "x", "left"])
        h    = _get_field(rec, ["h", "height"])
        w    = _get_field(rec, ["w", "width"])
        if name is None or r0 is None or c0 is None or h is None or w is None:
            return
        base = os.path.splitext(os.path.basename(_to_str_name(name)))[0]
        recs.append({
            "id": base,
            "row0": int(np.array(r0).ravel()[0]),
            "col0": int(np.array(c0).ravel()[0]),
            "h": int(np.array(h).ravel()[0]),
            "w": int(np.array(w).ravel()[0])
        })

    if isinstance(container, np.ndarray):
        for e in np.atleast_1d(container).ravel():
            _push(e)
    elif isinstance(container, list):
        for e in container:
            _push(e)
    else:
        raise ValueError("Unsupported patch_index format.")

    if not recs:
        raise ValueError("No valid records extracted from patch_index.mat")
    return recs

# ============================== File map (stitching) ==============================

def _digits(s): return "".join([c for c in s if c.isdigit()])

def _strip_leading_zeros(d):
    if not d: return d
    d2 = d.lstrip("0")
    return d2 if d2 != "" else "0"

def _norm_keys_from_basename(base):
    b = base.lower()
    for p in ["patch_", "tile_", "img_", "image_", "slice_", "patch", "tile", "img", "image", "slice"]:
        if b.startswith(p):
            b = b[len(p):]
            break
    d = _digits(b)
    dz = _strip_leading_zeros(d)
    return base.lower(), d, dz

def build_file_map(input_dir):
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    map_exact = {}
    map_norm = {}
    all_files = []
    for f in os.listdir(input_dir):
        base, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        path = os.path.join(input_dir, f)
        base_l = base.lower()
        if base_l in map_exact:
            if map_exact[base_l].lower().endswith(".png"):
                pass
            elif ext.lower() == ".png":
                map_exact[base_l] = path
        else:
            map_exact[base_l] = path

        b_key, d_key, dz_key = _norm_keys_from_basename(base)
        for k in {b_key, d_key, dz_key}:
            if not k:
                continue
            map_norm.setdefault(k, []).append(path)

        all_files.append((b_key, d_key, dz_key, path))

    def _order_key(t):
        _, _, dz, p = t
        if dz.isdigit():
            return (0, int(dz), p.lower())
        return (1, p.lower())

    all_files_sorted = [t[3] for t in sorted(all_files, key=_order_key)]
    return map_exact, map_norm, all_files_sorted

def _find_tile_path(base, map_exact, map_norm, all_files_sorted, rec_index=None):
    b_key = os.path.splitext(os.path.basename(base))[0].lower()
    if b_key in map_exact:
        return map_exact[b_key], "exact"
    _, dkey, dzkey = _norm_keys_from_basename(b_key)
    for k in [dkey, dzkey, b_key]:
        if k and k in map_norm:
            cands = map_norm[k]
            if len(cands) == 1:
                return cands[0], f"norm({k})"
            for p in cands:
                if os.path.splitext(os.path.basename(p))[0].lower() == b_key:
                    return p, f"norm-exact({k})"
            return sorted(cands)[0], f"norm-first({k})"
    if rec_index is not None and 0 <= rec_index < len(all_files_sorted):
        return all_files_sorted[rec_index], "fallback-by-order"
    return None, "miss"

# ============================== Stitching (first-write-wins) ==============================

def stitch_saliency_first_write(recs: List[Dict[str, Any]], map_exact, map_norm, all_files_sorted,
                                H: int, W: int, thr: float = 0.5, out_dir: Optional[str] = None):
    sal_full = np.zeros((H, W), dtype=np.float64)
    w_full   = np.zeros((H, W), dtype=np.float32)
    filled   = np.zeros((H, W), dtype=bool)

    miss = 0
    used = 0
    miss_examples = []

    for i, rec in enumerate(recs):
        base = rec["id"]
        path, _ = _find_tile_path(base, map_exact, map_norm, all_files_sorted, rec_index=i)
        if path is None:
            miss += 1
            if len(miss_examples) < 20:
                miss_examples.append(base)
            continue

        img = skio.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        S = img.astype(np.float64)
        if S.max() > 1.0:
            S = S / 255.0

        ph, pw = int(rec["h"]), int(rec["w"])
        if S.shape != (ph, pw):
            S = resize(S, (ph, pw), mode='reflect', anti_aliasing=False, preserve_range=True)
        S = np.clip(S, 0.0, 1.0)

        r0, c0 = int(rec["row0"]), int(rec["col0"])
        r1 = min(r0 + ph, H)
        c1 = min(c0 + pw, W)
        if r0 >= H or c0 >= W or r1 <= r0 or c1 <= c0:
            continue

        tile_h = r1 - r0
        tile_w = c1 - c0
        S = S[:tile_h, :tile_w]

        overlap = filled[r0:r1, c0:c1]
        new_mask = ~overlap
        if not new_mask.any():
            continue

        view = sal_full[r0:r1, c0:c1]
        view[new_mask] = S[new_mask]
        filled[r0:r1, c0:c1] |= new_mask
        w_full[r0:r1, c0:c1] += new_mask
        used += 1

    eps = 1e-8
    sal_full = sal_full / (w_full + eps)
    sal_full = np.clip(sal_full, 0.0, 1.0)
    sal_mask = (sal_full >= float(thr))

    stats = {"tiles_total": len(recs), "tiles_used": used, "tiles_missing": miss}
    if miss > 0 and out_dir is not None:
        dbg = os.path.join(out_dir, "missing_tiles_debug.txt")
        with open(dbg, "w", encoding="utf-8") as f:
            f.write(f"Total recs: {len(recs)}, used: {used}, missing: {miss}\n")
            f.write("Examples of missing rec IDs (up to 20):\n")
            for b in miss_examples:
                f.write(f"  - {b}\n")
    return sal_full.astype(np.float32), sal_mask.astype(np.uint8), stats

# ============================== 2D CA-CFAR ==============================

def cfar_2d(P, Pfa=1e-6, Gr=8, Gd=1, Rr=12, Rd2=4, roi_mask=None):
    Nd, Nr = P.shape
    blkH = 2*(Rd2+Gd)+1
    blkW = 2*(Rr+Gr)+1
    gH   = 2*Gd+1
    gW   = 2*Gr+1
    numRef = blkH*blkW - gH*gW
    alpha  = numRef * (Pfa**(-1.0/numRef) - 1.0)

    rStart = Rr + Gr
    rEnd   = Nr - (Rr + Gr)
    dStart = Rd2 + Gd
    dEnd   = Nd - (Rd2 + Gd)

    det = np.zeros((Nd, Nr), dtype=bool)
    thr = np.full((Nd, Nr), np.nan, dtype=np.float64)

    for di in range(dStart, dEnd):
        d_lo = di - (Rd2 + Gd)
        d_hi = di + (Rd2 + Gd) + 1
        dg_lo = di - Gd
        dg_hi = di + Gd + 1
        for ri in range(rStart, rEnd):
            if roi_mask is not None and not roi_mask[di, ri]:
                continue
            r_lo = ri - (Rr + Gr)
            r_hi = ri + (Rr + Gr) + 1
            rg_lo = ri - Gr
            rg_hi = ri + Gr + 1

            block = P[d_lo:d_hi, r_lo:r_hi]
            block_guard = np.ones_like(block, dtype=bool)
            block_guard[dg_lo - d_lo: dg_hi - d_lo, rg_lo - r_lo: rg_hi - r_lo] = False
            ref_vals = block[block_guard]
            if ref_vals.size == 0:
                continue

            noise_est = ref_vals.mean()
            T = alpha * noise_est
            thr[di, ri] = T
            det[di, ri] = (P[di, ri] > T)

    return det, thr

# ============================== Detection export helpers ==============================

def export_detections(savedir: str, det_mask: np.ndarray, P: np.ndarray,
                      r_axis: np.ndarray, fd_axis: np.ndarray,
                      sort_by: str = "power") -> int:
    """
    Export a CSV and MAT file of detections where det_mask==True.
    Columns/fields: row, col, range_m, fd_Hz, power_linear, power_dB
    """
    rows, cols = np.where(det_mask)
    if rows.size == 0:
        # still create empty CSV/MAT with header
        csv_path = os.path.join(savedir, "detections.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row", "col", "range_m", "fd_Hz", "power_linear", "power_dB"])
        sio.savemat(os.path.join(savedir, "detections.mat"),
                    {"row": np.array([], dtype=np.int32),
                     "col": np.array([], dtype=np.int32),
                     "range_m": np.array([], dtype=np.float64),
                     "fd_Hz": np.array([], dtype=np.float64),
                     "power_linear": np.array([], dtype=np.float64),
                     "power_dB": np.array([], dtype=np.float64)})
        return 0

    dets = []
    eps = 1e-12
    for di, ri in zip(rows.tolist(), cols.tolist()):
        p_lin = float(P[di, ri])
        p_db  = 10.0 * np.log10(max(p_lin, eps))
        rng   = float(r_axis[ri]) if ri < r_axis.size else float("nan")
        fd    = float(fd_axis[di]) if di < fd_axis.size else float("nan")
        
        # [SANITY] dB 与线性值互相可还原的断言（插在这里）
        p_lin_back = 10.0 ** (p_db / 10.0)
        if (not np.isfinite(p_lin)) or (not np.isfinite(p_db)) or \
           (abs(p_lin_back - p_lin) > max(1e-9 * max(1.0, p_lin), 1e-12)):
            raise RuntimeError(
                f"Inconsistent lin/dB at (row={di}, col={ri}): "
                f"lin={p_lin}, dB={p_db}, back={p_lin_back}"
            )
        dets.append((di, ri, rng, fd, p_lin, p_db))

    if sort_by == "power":
        dets.sort(key=lambda x: x[4], reverse=True)  # power_linear desc
    elif sort_by == "rowcol":
        dets.sort(key=lambda x: (x[0], x[1]))

    # CSV
    csv_path = os.path.join(savedir, "detections.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "range_m", "fd_Hz", "power_linear", "power_dB"])
        w.writerows(dets)

    # MAT
    sio.savemat(os.path.join(savedir, "detections.mat"),
                {"row": np.array([d[0] for d in dets], dtype=np.int32),
                 "col": np.array([d[1] for d in dets], dtype=np.int32),
                 "range_m": np.array([d[2] for d in dets], dtype=np.float64),
                 "fd_Hz": np.array([d[3] for d in dets], dtype=np.float64),
                 "power_linear": np.array([d[4] for d in dets], dtype=np.float64),
                 "power_dB": np.array([d[5] for d in dets], dtype=np.float64)})

    return len(dets)

def save_overlay(savedir: str, P: np.ndarray, det_mask: np.ndarray, out_name: str = "cfar_det_overlay.png"):
    """
    Save an overlay image: background = percentile-stretched power map,
    detections = red dots (3x3)
    """
    bg = P.astype(np.float64)
    eps = 1e-12
    # dB for visualization (optional), then linear stretch 2-98 percentile
    bg_db = 10.0 * np.log10(np.maximum(bg, eps))
    lo, hi = np.percentile(bg_db, [2, 98])
    if hi <= lo:
        lo, hi = bg_db.min(), bg_db.max()
    norm = (bg_db - lo) / (hi - lo + 1e-9)
    gray = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)  # H,W,3

    ys, xs = np.where(det_mask)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 1), min(rgb.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(rgb.shape[1], x + 2)
        rgb[y0:y1, x0:x1, 0] = 255  # R
        rgb[y0:y1, x0:x1, 1] = 0    # G
        rgb[y0:y1, x0:x1, 2] = 0    # B

    skio.imsave(os.path.join(savedir, out_name), rgb)

# ============================== Main ==============================

def main():
    ap = argparse.ArgumentParser(description="Author-style SDNet saliency (dataloader_sod) + stitch + CFAR + detections export")

    # Saliency I/O (author-style)
    ap.add_argument("--datadir",  required=True, type=str, help="root dir containing {testdata}/images")
    ap.add_argument("--testdata", required=True, type=str, help="dataset name under datadir (expects datadir/testdata/images)")
    ap.add_argument("--savedir",  required=True, type=str, help="where to save saliency PNGs and results")
    ap.add_argument("--evaluate", required=True, type=str, help="checkpoint path (e.g., checkpoints/sdneta_from_pretrained.pth)")
    ap.add_argument("--model", type=str, default="sdneta", help="model entry (e.g., sdnet, sdneta)")
    ap.add_argument("--bn", action="store_true", help="use BN in backbone for training-time model (author style)")
    ap.add_argument("--train-config", type=str, default="sdnet-a", help="training-time model config")
    ap.add_argument("--inference-config", type=str, default="baseline", help="inference-time model config")
    ap.add_argument("--gpu", type=str, default="", help="CUDA_VISIBLE_DEVICES id(s), empty means CPU if no CUDA")
    ap.add_argument("--size", type=int, default=None, help="resize to (size,size); None keeps original")

    ap.add_argument("-j", "--workers", type=int, default=4, help="workers for saliency dataloader")

    # Stitch & CFAR
    ap.add_argument("--patch_index", required=True, help="patch_index.mat (tile placement metadata)")
    ap.add_argument("--meta",        required=True, help="rd_meta.mat")
    ap.add_argument("--rd_power",    required=True, help="rd_power.mat")
    ap.add_argument("--thr", type=float, default=0.5, help="threshold on stitched saliency for ROI")
    ap.add_argument("--Pfa", type=float, default=1e-6, help="CFAR false-alarm probability")

    args = ap.parse_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = torch.cuda.is_available()

    # Build models (author-style)
    args_m_tr = argparse.Namespace(model=args.model, config=args.train_config, bn=bool(args.bn))
    model_training_time = getattr(models, args.model)(args_m_tr)

    args_m_inf = argparse.Namespace(model=args.model, config=args.inference_config, bn=False)
    model_inference_time = getattr(models, args.model)(args_m_inf)

    if use_cuda:
        model_inference_time = torch.nn.DataParallel(model_inference_time).cuda()
        model_training_time = torch.nn.DataParallel(model_training_time).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    # Load checkpoint (CPU/GPU compatible)
    print("=> loading checkpoint from '{}'".format(args.evaluate))
    checkpoint_dict = torch.load(args.evaluate, map_location='cpu')
    state = checkpoint_dict["state_dict"] if (isinstance(checkpoint_dict, dict) and "state_dict" in checkpoint_dict) else checkpoint_dict

    if not use_cuda:
        # Strip leading 'module.' for CPU model
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state[new_k] = v
        state = new_state

    model_training_time.load_state_dict(state)
    print("=> loaded checkpoint '{}' successfully; begin to reparameterize".format(args.evaluate))

    # DCR (author-style)
    if use_cuda:
        model_training_time.module.reparameterize()
    else:
        model_training_time.reparameterize()
    state_from_training(model_training_time, model_inference_time)
    print("=> reparameterization done")

    # Prepare saliency dataloader (author-style)
    args_sal = argparse.Namespace(
        savedir=args.savedir,
        datadir=args.datadir,
        testdata=args.testdata,
        evaluate=args.evaluate,
        model=args.model,
        bn=bool(args.bn),
        train_config=args.train_config,
        inference_config=args.inference_config,
        gpu=args.gpu,
        size=(args.size, args.size) if args.size is not None else None,
        workers=args.workers,
        use_cuda=use_cuda,
    )
    test_loader = get_saldata(args_sal)

    # Output dir for saliency PNGs: savedir/testdata
    sal_dir = os.path.join(args.savedir, args.testdata)
    if not os.path.exists(sal_dir):
        os.makedirs(sal_dir)
    else:
        print('%s already exists, but it will be regenerated' % sal_dir)

    # Saliency inference (author-style loop)
    model_inference_time.eval()
    print(f"\nBegin to generating saliency maps for model {args.model}...\nImg generated in {sal_dir}\n")
    t0 = time.time()
    for idx, (image, img_name, imgsize) in enumerate(test_loader):
        img_name = img_name[0]
        imgsize = [x.item() for x in imgsize]
        with torch.no_grad():
            image = image.cuda() if use_cuda else image
            result = model_inference_time(image)
            result = F.interpolate(result, mode='bilinear', size=imgsize, align_corners=False)
            result = torch.squeeze(result).cpu().numpy()
        result_img = Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
        result_img.save(os.path.join(sal_dir, f"{img_name}.png"))
        if idx % 100 == 0:
            print(f"Running test [{idx + 1}/{len(test_loader)}]")
    t1 = time.time()
    print(f"[OK] Saliency generated. Time: {t1 - t0:.2f}s; dir={sal_dir}")

    # Load RD meta / power
    Nd, Nr, r_axis, fd_axis, extras = load_meta(args.meta)
    print(f"[INFO] meta: Nd={Nd}, Nr={Nr}, r_axis={r_axis.size}, fd_axis={fd_axis.size}")
    P = load_power(args.rd_power, Nd_expected=Nd, Nr_expected=Nr)
    print(f"[INFO] power: P.shape={P.shape}")
   
    probe_r, probe_c = 632, 561
    val = float(P[probe_r, probe_c])
    val_db = 10.0*np.log10(max(val, 1e-12))
    print(f"[SANITY] P[{probe_r},{probe_c}] = {val:.12g}  => {val_db:.6f} dB")

    mi = int(np.argmax(P))
    mi_r, mi_c = np.unravel_index(mi, P.shape)
    vmax = float(P[mi_r, mi_c])
    vmax_db = 10.0*np.log10(max(vmax, 1e-12))
    print(f"[SANITY] GLOBAL MAX at (row={mi_r}, col={mi_c}) = {vmax:.12g}  => {vmax_db:.6f} dB")
    
    # Stitch from saliency folder
    recs = load_patch_rec(args.patch_index)
    map_exact, map_norm, all_files_sorted = build_file_map(sal_dir)
    sal_full, sal_mask, st = stitch_saliency_first_write(
        recs, map_exact, map_norm, all_files_sorted, H=Nd, W=Nr, thr=args.thr, out_dir=args.savedir
    )
    print(f"[STITCH] used={st['tiles_used']} / total={st['tiles_total']} (missing={st['tiles_missing']})")

    # CFAR within ROI
    assert sal_mask.shape == P.shape, f"ROI mask shape {sal_mask.shape} != power shape {P.shape}"
    det_mask, thr_map = cfar_2d(P, Pfa=args.Pfa, roi_mask=sal_mask.astype(bool))

    # Save base maps as before
    skio.imsave(os.path.join(args.savedir, "cfar_det_mask.png"), (det_mask.astype(np.uint8) * 255))
    np.save(os.path.join(args.savedir, "cfar_thr_map.npy"), thr_map.astype(np.float32))
    skio.imsave(os.path.join(args.savedir, "saliency_full_uint8.png"), (np.clip(sal_full, 0, 1) * 255).astype(np.uint8))
    skio.imsave(os.path.join(args.savedir, "saliency_mask_bin.png"), (sal_mask.astype(np.uint8) * 255))

    # NEW: Export detections list and overlay
    n_det = export_detections(args.savedir, det_mask, P, r_axis, fd_axis, sort_by="power")
    save_overlay(args.savedir, P, det_mask, out_name="cfar_det_overlay.png")
    print(f"[DETECTIONS] exported {n_det} detections to detections.csv and detections.mat")
    print(f"[OK] CFAR detection completed and saved in {args.savedir}")

if __name__ == "__main__":
    main()