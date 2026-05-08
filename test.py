"""
test.py
-------
Inference script for the cluster-shared-encoder + personal-ensemble pipeline.

For each patient:
  1. Load cluster assignment → load the correct shared encoder
  2. Load patient's personal ensemble head
  3. Load patient's personal train_dist (anomaly reference)
  4. For each val/test day: compute windowed ensemble variance → raw a_norm score
  5. Apply temporal aggregation over the previous N days (per-patient lookback)
  6. Apply per-patient threshold to binarise the aggregated score
  7. Compute AUROC / AUPRC (val mode) or write submission CSV (test mode)

Temporal Aggregation Modes
---------------------------
Each day first produces a continuous normalised anomaly score  a_norm.
Before thresholding, the score is smoothed over the last `lookback` days
(using however many days are available at the start of an episode).
The window resets at the beginning of every episode subfolder.

  --agg_mode A  Rolling mean of a_norm         (smooth, continuous)
  --agg_mode B  Rolling mean of binary flags    (majority-vote style)
  --agg_mode C  Recency-weighted mean of a_norm (exponential decay, recent > old)
  --agg_mode D  Rolling max of a_norm           (sensitive, high-recall)

Per-patient lookback
---------------------
Pass --lookbacks with exactly --num_patients integer values (one per patient,
ordered P1 … P<num_patients>).  Default: 4 for every patient.

Per-patient thresholds
----------------------
Pass --thresholds with exactly --num_patients float values.
The aggregated score is compared against the patient threshold:
  final_score = 1.0  if  agg_score > threshold_i  else  0.0
Default: 0.0 for every patient.

Examples
--------
# Val — rolling mean, 4-day lookback for all patients
python test.py \\
    --features_path data/track_2_new_features/ \\
    --dataset_path  data/track_2/ \\
    --load_path     checkpoints_clustered/ \\
    --mode          val \\
    --agg_mode      A \\
    --lookbacks     4 4 4 4 4 4 4 4

# Val — recency-weighted, mixed per-patient lookback and thresholds
python test.py \\
    --features_path data/track_2_new_features/ \\
    --dataset_path  data/track_2/ \\
    --load_path     checkpoints_clustered/ \\
    --mode          val \\
    --agg_mode      C \\
    --lookbacks     4 7 4 4 7 4 7 4 \\
    --thresholds    0.0 0.1 -0.05 0.2 0.0 0.15 -0.1 0.3

# Test — rolling max, 7-day lookback for all patients
python test.py \\
    --features_path data/track_2_new_features/ \\
    --dataset_path  data/track_2/ \\
    --load_path     checkpoints_clustered/ \\
    --mode          test \\
    --agg_mode      D \\
    --lookbacks     7 7 7 7 7 7 7 7
"""

from pprint import pprint
import argparse, os, pickle
import numpy as np
import pandas as pd
import torch
import sklearn.metrics

from model import TransformerHeartPredictor
from trainer import create_ensemble_mlp

COLS8 = [
    "acc_norm",
    "gyr_norm",
    "heartRate_mean",
    "rRInterval_mean",
    "rRInterval_rmssd",
    "rRInterval_sdnn",
    "rRInterval_lombscargle_power_high",
    "steps",
]
COLS10 = COLS8 + ["sin_t", "cos_t"]


# -------------------------------------------------------
# Time helpers
# -------------------------------------------------------
def calculate_sincos_from_minutes(minutes):
    ang = minutes * (2.0 * np.pi / (60 * 24))
    return np.sin(ang), np.cos(ang)


def ensure_time_cols(df):
    if ("sin_t" not in df.columns) or ("cos_t" not in df.columns):
        if "mins" not in df.columns:
            raise ValueError("Missing 'mins' column to compute sin_t/cos_t.")
        sin_t, cos_t = calculate_sincos_from_minutes(df["mins"])
        df = df.copy()
        df["sin_t"] = sin_t
        df["cos_t"] = cos_t
    return df


def map_scaled_to_model(seq_scaled, cols_scaler, cols_model, raw_seq=None):
    idx_map = [cols_scaler.index(c) for c in cols_model if c in cols_scaler]
    out = seq_scaled[:, idx_map]
    missing = [c for c in cols_model if c not in cols_scaler]
    if missing and raw_seq is not None:
        extra = np.stack([raw_seq[:, COLS10.index(c)] for c in missing], axis=1)
        out = np.concatenate([out, extra], axis=1)
    return out


# -------------------------------------------------------
# Temporal aggregation helpers
# -------------------------------------------------------
def _exp_weights(n: int) -> np.ndarray:
    """
    Exponential decay weights for n days, oldest → newest.
    w_i = exp(i) / sum(exp(j) for j in 0..n-1)
    """
    w = np.exp(np.arange(n, dtype=np.float64))
    return w / w.sum()


def aggregate_score(
    history: list,          # list of raw a_norm floats (oldest … newest), episode-local
    lookback: int,          # max days to look back
    threshold: float,       # binarisation threshold (only used for mode B)
    mode: str,              # "A" | "B" | "C" | "D"
) -> float:
    """
    Given the running history of a_norm values for the current episode
    (including today's value as the last element), return the aggregated score.

    - Uses min(lookback, len(history)) days so the start of an episode is fine.
    - Resets across episodes because `history` is created fresh per episode.
    """
    window = history[-lookback:]          # shrinks naturally at episode start
    n = len(window)
    arr = np.array(window, dtype=np.float64)

    if mode == "A":
        # Rolling mean of raw a_norm scores
        return float(np.mean(arr))

    elif mode == "B":
        # Rolling mean of binary decisions (majority-vote style)
        binary = (arr > threshold).astype(np.float64)
        return float(np.mean(binary))

    elif mode == "C":
        # Recency-weighted mean (exponential decay, recent days count more)
        w = _exp_weights(n)
        return float(np.dot(w, arr))

    elif mode == "D":
        # Rolling max — flag today if any day in window was high
        return float(np.max(arr))

    else:
        raise ValueError(f"Unknown aggregation mode '{mode}'. Choose A, B, C, or D.")


# -------------------------------------------------------
# Argument Parser
# -------------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--window_size",      type=int,   default=24)
    p.add_argument("--input_features",   type=int,   default=8)
    p.add_argument("--output_dim",       type=int,   default=5)
    p.add_argument("--d_model",          type=int,   default=64)
    p.add_argument("--nhead",            type=int,   default=8)
    p.add_argument("--nlayers",          type=int,   default=2)
    p.add_argument("--ensembles",        type=int,   default=5)
    p.add_argument("--num_patients",     type=int,   default=8)

    p.add_argument("--features_path",   type=str, required=True)
    p.add_argument("--dataset_path",    type=str, required=True)
    p.add_argument("--submission_path", type=str, default="/var/tmp/spgc-submission")
    p.add_argument("--load_path",       type=str, required=True,
                   help="Directory containing per-patient checkpoint subdirs AND "
                        "cluster_assignments.pkl / cluster_groups.pkl")
    p.add_argument("--device",          type=str, default="cpu")
    p.add_argument("--mode",            type=str, default="test",
                   choices=["val", "test"])

    # ------------------------------------------------------------------
    # Temporal aggregation
    # ------------------------------------------------------------------
    p.add_argument(
        "--agg_mode",
        type=str,
        default="A",
        choices=["A", "B", "C", "D"],
        help=(
            "Temporal aggregation strategy applied over the lookback window:\n"
            "  A — rolling mean of raw a_norm  (smooth, continuous)\n"
            "  B — rolling mean of binary flags (majority-vote; uses threshold)\n"
            "  C — recency-weighted mean of a_norm (exponential decay)\n"
            "  D — rolling max of a_norm  (sensitive, high-recall)\n"
            "Default: A"
        ),
    )

    p.add_argument(
        "--lookbacks",
        type=int,
        nargs="+",
        default=None,
        metavar="L",
        help=(
            "Per-patient lookback window in days, ordered P1 … P<num_patients>. "
            "Must supply exactly --num_patients values. "
            "Defaults to 4 for all patients if omitted."
        ),
    )

    # ------------------------------------------------------------------
    # Per-patient thresholds
    # ------------------------------------------------------------------
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Per-patient anomaly thresholds, one float per patient ordered "
            "P1 … P<num_patients>. Applied to the AGGREGATED score. "
            "For mode B the threshold is also used internally to binarise "
            "each day before averaging. "
            "Defaults to 0.0 for all patients if omitted."
        ),
    )

    args = p.parse_args()
    args.seq_len = args.window_size

    # Validate / build lookback list
    if args.lookbacks is None:
        args.lookbacks = [4] * args.num_patients
    else:
        if len(args.lookbacks) != args.num_patients:
            p.error(
                f"--lookbacks requires exactly {args.num_patients} values "
                f"(got {len(args.lookbacks)})."
            )

    # Validate / build threshold list
    if args.thresholds is None:
        args.thresholds = [0.0] * args.num_patients
    else:
        if len(args.thresholds) != args.num_patients:
            p.error(
                f"--thresholds requires exactly {args.num_patients} values "
                f"(got {len(args.thresholds)})."
            )

    return args


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    args = parse()
    pprint(vars(args))
    device = args.device

    print(f"\nAggregation mode : {args.agg_mode}")
    print("Per-patient settings:")
    for i in range(args.num_patients):
        print(f"  P{i + 1}: lookback={args.lookbacks[i]}  threshold={args.thresholds[i]}")
    print()

    # ===========================================================
    # Load cluster assignments
    # ===========================================================
    assignments_path = os.path.join(args.load_path, "cluster_assignments.pkl")
    groups_path      = os.path.join(args.load_path, "cluster_groups.pkl")

    if not os.path.exists(assignments_path):
        raise FileNotFoundError(
            f"cluster_assignments.pkl not found in {args.load_path}. "
            "Run train.py first."
        )

    with open(assignments_path, "rb") as f:
        cluster_assignments = pickle.load(f)
    with open(groups_path, "rb") as f:
        groups = pickle.load(f)

    num_clusters = len(groups)
    print(f"Loaded clusters: {groups}")

    # ===========================================================
    # Load one shared encoder per cluster
    # ===========================================================
    cluster_encoders = {}
    cluster_scalers  = {}

    for c_id, members in groups.items():
        representative = sorted(members)[0]
        rep_id = int(representative[1:])
        pdir   = os.path.join(args.load_path, str(rep_id))

        enc_pth    = os.path.join(pdir, "best_encoder.pth")
        scaler_pth = os.path.join(pdir, "scaler.pkl")

        if not os.path.exists(enc_pth):
            print(f"Warning: encoder not found for cluster {c_id} at {enc_pth}, skipping cluster.")
            cluster_encoders[c_id] = None
            cluster_scalers[c_id]  = None
            continue

        state    = torch.load(enc_pth, map_location="cpu")
        in_feats = state["encoder_input_layer.weight"].shape[1]

        local_args = vars(args).copy()
        local_args["input_features"] = in_feats
        local_args["device"]         = device

        model = TransformerHeartPredictor(local_args).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        cluster_encoders[c_id] = model

        with open(scaler_pth, "rb") as f:
            cluster_scalers[c_id] = pickle.load(f)

        print(f"  Loaded shared encoder for cluster {c_id} (representative: {representative})")

    # ===========================================================
    # Load per-patient ensemble heads, train_dists, col config
    # ===========================================================
    personal_heads = {}
    personal_dists = {}
    personal_cols  = {}

    for pid in range(args.num_patients):
        patient_str = f"P{pid + 1}"
        pdir        = os.path.join(args.load_path, str(pid + 1))

        ens_pth  = os.path.join(pdir, "best_ensembles.pth")
        dist_pth = os.path.join(pdir, "train_dist_anomaly_scores.pkl")

        if not (os.path.exists(ens_pth) and os.path.exists(dist_pth)):
            print(f"Warning: missing artifacts for {patient_str}, skipping.")
            personal_heads[pid] = None
            personal_dists[pid] = None
            personal_cols[pid]  = None
            continue

        ensemble = create_ensemble_mlp(args)
        ensemble.load_state_dict(torch.load(ens_pth, map_location=device))
        ensemble.eval()
        personal_heads[pid] = ensemble

        with open(dist_pth, "rb") as f:
            td = pickle.load(f)
            personal_dists[pid] = td[pid]

        c_id    = cluster_assignments[patient_str]
        encoder = cluster_encoders.get(c_id)
        if encoder is not None:
            state    = torch.load(
                os.path.join(args.load_path, str(pid + 1), "best_encoder.pth"),
                map_location="cpu"
            )
            in_feats = state["encoder_input_layer.weight"].shape[1]
            personal_cols[pid] = COLS10 if in_feats == 10 else COLS8
        else:
            personal_cols[pid] = COLS8

    torch.set_grad_enabled(False)
    all_auroc, all_auprc = [], []

    # ===========================================================
    # Iterate over patients
    # ===========================================================
    for patient in sorted(os.listdir(args.features_path)):
        if patient == ".DS_Store":
            continue

        pid         = int(patient[1:]) - 1
        patient_str = f"P{pid + 1}"

        if pid not in personal_heads or personal_heads[pid] is None:
            continue
        if patient_str not in cluster_assignments:
            continue

        c_id    = cluster_assignments[patient_str]
        encoder = cluster_encoders.get(c_id)
        if encoder is None:
            continue

        scaler     = cluster_scalers[c_id]
        head       = personal_heads[pid]
        train_dist = personal_dists[pid]
        cols_model = personal_cols[pid]

        # Per-patient settings
        patient_threshold = args.thresholds[pid]
        patient_lookback  = args.lookbacks[pid]
        print(
            f"{patient_str}: threshold={patient_threshold}  "
            f"lookback={patient_lookback}  agg={args.agg_mode}"
        )

        n_scaler    = getattr(scaler, "n_features_in_", 8)
        cols_scaler = COLS10 if n_scaler == 10 else COLS8

        _mean = float(np.mean(train_dist))
        _max  = float(np.max(train_dist))
        _min  = float(np.min(train_dist))
        denom = (_max - _min) if (_max - _min) != 0 else 1.0

        pdir = os.path.join(args.features_path, patient)
        user_preds, user_labels = [], []

        # ----------------------------------------------------------
        # Iterate over episode subfolders
        # The a_norm history resets at the start of each episode.
        # ----------------------------------------------------------
        for sub in sorted(os.listdir(pdir)):
            if not (
                (args.mode == "val"  and "val"  in sub and sub.endswith("val")) or
                (args.mode == "test" and "test" in sub)
            ):
                continue

            fpath = os.path.join(pdir, sub, "features_stretched_w_steps.csv")
            if not os.path.exists(fpath):
                continue

            df = pd.read_csv(fpath).replace([np.inf, -np.inf], np.nan).dropna()
            df = ensure_time_cols(df)

            if args.mode == "test":
                relapse_df = pd.read_csv(
                    os.path.join(args.dataset_path, patient, sub, "relapses.csv")
                )
                relapse_df = relapse_df.iloc[:-1]
                DAY_INDEX  = "day_index"
            else:
                relapse_df = pd.read_csv(os.path.join(pdir, sub, "relapse_stretched.csv"))
                DAY_INDEX  = "day"

            # ---- Episode-local a_norm history (resets here) ----
            episode_anorm_history: list = []

            for day_idx in relapse_df[DAY_INDEX].unique():
                day_df = df[df[DAY_INDEX] == day_idx]

                if len(day_df) < args.window_size + 1:
                    # Not enough data — treat as neutral (0.0) raw score
                    raw_anorm = 0.0
                else:
                    sequences = []
                    step      = max(1, args.window_size // 3)
                    starts    = (
                        [0]
                        if len(day_df) == args.window_size + 1
                        else range(0, len(day_df) - args.window_size, step)
                    )

                    for s in starts:
                        seq_raw    = day_df.iloc[s: s + args.window_size][cols_scaler].to_numpy()
                        seq_scaled = scaler.transform(seq_raw)
                        raw_seq    = day_df.iloc[s: s + args.window_size][COLS10].to_numpy()
                        seq_model  = map_scaled_to_model(seq_scaled, cols_scaler, cols_model, raw_seq)
                        sequences.append(seq_model)

                    sequence   = np.stack(sequences)
                    seq_tensor = (
                        torch.tensor(sequence, dtype=torch.float32)
                        .permute(0, 2, 1)
                        .to(device)
                    )

                    k           = args.ensembles
                    features, _ = encoder(seq_tensor)
                    bf          = features[None, :, :].repeat([k, 1, 1])
                    preds       = head(bf)

                    avg      = torch.mean(preds, 0)
                    var      = torch.sum((preds - avg) ** 2, dim=2)
                    mean_var = torch.mean(torch.mean(var, 0)).item()

                    raw_anorm = (mean_var - _mean) / denom

                # ---- Append to episode history, then aggregate ----
                episode_anorm_history.append(raw_anorm)

                agg_score = aggregate_score(
                    history   = episode_anorm_history,
                    lookback  = patient_lookback,
                    threshold = patient_threshold,   # used internally by mode B
                    mode      = args.agg_mode,
                )

                # ---- Final binary decision ----
                # Mode B already produces a [0,1] vote ratio; we still threshold it
                # (e.g. >0.5 means majority positive).
                a = 1.0 if agg_score > patient_threshold else 0.0

                relapse_df.loc[relapse_df[DAY_INDEX] == day_idx, "score"] = a
                user_preds.append(a)
                if "relapse" in relapse_df.columns:
                    lbl = relapse_df[relapse_df[DAY_INDEX] == day_idx]["relapse"].to_numpy()[0]
                    user_labels.append(lbl)

            if args.mode == "test":
                save_dir = os.path.join(args.submission_path, f"patient{patient[1]}", sub)
                os.makedirs(save_dir, exist_ok=True)
                relapse_df.to_csv(os.path.join(save_dir, "submission.csv"), index=False)
                print(f"  Saved submission → {save_dir}")

        # ---- Metrics (val mode) ----
        if user_labels and len(np.unique(user_labels)) > 1:
            y_true       = np.array(user_labels)
            y_pred       = np.array(user_preds)
            fpr, tpr, _  = sklearn.metrics.roc_curve(y_true, y_pred)
            prec, rec, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred)
            auroc        = sklearn.metrics.auc(fpr, tpr)
            auprc        = sklearn.metrics.auc(rec, prec)
            print(
                f"{patient}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                f"AVG={(auroc + auprc) / 2:.4f}"
            )
            all_auroc.append(auroc)
            all_auprc.append(auprc)
        else:
            print(f"{patient}: skipped metrics (labels missing or constant).")

    if all_auroc:
        total_auroc = float(np.mean(all_auroc))
        total_auprc = float(np.mean(all_auprc))
        total_avg   = (total_auroc + total_auprc) / 2
        print(
            f"\nTOTAL  AUROC={total_auroc:.4f}  AUPRC={total_auprc:.4f}  "
            f"AVG={total_avg:.4f}"
        )


if __name__ == "__main__":
    main()
