"""
train.py
--------
Entry point for the cluster-shared-encoder + personal-ensemble pipeline.

Steps
-----
0. Cluster patients (or load existing clusters from --save_path)
1. Build ClusterDataset (one per cluster) for shared encoder training
2. Build PatientDataset (one per patient) for personal head training / val
3. Build dataloaders
4. Instantiate one TransformerHeartPredictor per cluster
5. Train via ClusterTrainer
"""

import os
import pickle
import argparse

import torch
from torch.optim.lr_scheduler import MultiStepLR

from model import TransformerHeartPredictor
from dataset import ClusterDataset, PatientDataset
from trainer import ClusterTrainer
from cluster_utils import build_clusters


# -------------------------------------------------------
# Device helper
# -------------------------------------------------------
def get_device(device_str="auto") -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.device(device_str)
        return device_str
    except Exception as e:
        print("Device error:", e)
        return "cpu"


# -------------------------------------------------------
# Argument Parser
# -------------------------------------------------------
def parse():
    parser = argparse.ArgumentParser(
        description="Train cluster-shared encoder + personal ensemble heads"
    )

    # Compute
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    parser.add_argument("--ensembles", type=int, default=5)

    # Transformer
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--input_features", type=int, default=8)
    parser.add_argument("--output_dim", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward_encoder", type=int, default=2048)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=2)

    # Patients / Clusters
    parser.add_argument("--num_patients", type=int, default=8)
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of patient clusters (inspect dendrogram to choose)")

    # Paths
    parser.add_argument("--features_path", default="data/track_2_new_features/", type=str)
    parser.add_argument("--dataset_path", default="data/track_2/", type=str)
    parser.add_argument("--save_path", type=str, default="checkpoints_clustered")

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--head_learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)

    default_device = get_device()
    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()
    args.seq_len = args.window_size
    return args


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    args = parse()
    device = args.device
    print(f"Using device: {device}")
    os.makedirs(args.save_path, exist_ok=True)

    # ===========================================================
    # Step 0: Cluster patients
    # ===========================================================
    assignments_path = os.path.join(args.save_path, "cluster_assignments.pkl")
    groups_path      = os.path.join(args.save_path, "cluster_groups.pkl")

    if os.path.exists(assignments_path) and os.path.exists(groups_path):
        print("Found existing cluster assignments — loading.")
        with open(assignments_path, "rb") as f:
            cluster_assignments = pickle.load(f)
        with open(groups_path, "rb") as f:
            groups = pickle.load(f)
    else:
        print("No existing clusters found — computing now.")
        cluster_assignments, groups = build_clusters(
            features_path=args.features_path,
            num_patients=args.num_patients,
            n_clusters=args.n_clusters,
            save_path=args.save_path,
        )

    num_clusters = len(groups)
    print(f"\nClusters: {groups}\n")

    # ===========================================================
    # Step 1: Build ClusterDatasets (shared encoder training)
    # ===========================================================
    print("Building ClusterDatasets for shared encoder training ...")
    cluster_train_datasets = {}
    cluster_scalers        = {}

    for c_id, members in groups.items():
        ds = ClusterDataset(
            features_path=args.features_path,
            dataset_path=args.dataset_path,
            patients=members,
            mode="train",
            window_size=args.window_size,
            stride=args.stride,
        )
        cluster_train_datasets[c_id] = ds
        cluster_scalers[c_id] = ds.scaler

        # Save cluster scaler — needed at test time
        scaler_path = os.path.join(args.save_path, f"cluster_{c_id}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(ds.scaler, f)
        print(f"  Saved cluster {c_id} scaler → {scaler_path}")

    # ===========================================================
    # Step 2: Build per-patient datasets (personal head + val)
    # ===========================================================
    print("\nBuilding per-patient datasets ...")
    patient_train_datasets     = {}
    patient_val_datasets       = {}
    patient_traindist_datasets = {}

    for pid in range(args.num_patients):
        patient_str = f"P{pid + 1}"
        c_id        = cluster_assignments[patient_str]
        scaler      = cluster_scalers[c_id]

        # Personal train dataset (uses cluster scaler — no refit)
        train_ds = PatientDataset(
            features_path=args.features_path,
            dataset_path=args.dataset_path,
            mode="train",
            scaler=scaler,
            window_size=args.window_size,
            stride=args.stride,
            patient=patient_str,
        )
        patient_train_datasets[pid] = train_ds

        # Save personal scaler (same as cluster scaler) for test.py compatibility
        pid_dir = os.path.join(args.save_path, str(pid + 1))
        os.makedirs(pid_dir, exist_ok=True)
        with open(os.path.join(pid_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        # Val dataset
        patient_val_datasets[pid] = PatientDataset(
            features_path=args.features_path,
            dataset_path=args.dataset_path,
            mode="val",
            scaler=scaler,
            window_size=args.window_size,
            stride=args.stride,
            patient=patient_str,
        )

        # Train-dist dataset (for anomaly reference distribution)
        patient_traindist_datasets[pid] = PatientDataset(
            features_path=args.features_path,
            dataset_path=args.dataset_path,
            mode="train",
            scaler=scaler,
            window_size=args.window_size,
            stride=args.stride,
            patient=patient_str,
        )

    # ===========================================================
    # Step 3: Dataloaders
    # ===========================================================

    # Cluster loaders (for shared encoder)
    cluster_loaders = {}
    for c_id in range(num_clusters):
        cluster_loaders[c_id] = {
            "train": torch.utils.data.DataLoader(
                cluster_train_datasets[c_id],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cores,
                pin_memory=True,
            )
        }

    # Patient loaders (for personal heads + val + train_dist)
    patient_loaders = {}
    for pid in range(args.num_patients):
        patient_loaders[pid] = {
            "train": torch.utils.data.DataLoader(
                patient_train_datasets[pid],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cores,
                pin_memory=True,
            ),
            "val": torch.utils.data.DataLoader(
                patient_val_datasets[pid],
                batch_size=1,
                shuffle=False,
                num_workers=args.cores,
                pin_memory=True,
            ),
            "train_dist": torch.utils.data.DataLoader(
                patient_traindist_datasets[pid],
                batch_size=1,
                shuffle=False,
                num_workers=args.cores,
                pin_memory=True,
            ),
        }

    # ===========================================================
    # Step 4: Models, optimizers, schedulers (one per cluster)
    # ===========================================================
    cluster_models = [
        TransformerHeartPredictor(vars(args)).to(device)
        for _ in range(num_clusters)
    ]

    n_params = sum(p.numel() for p in cluster_models[0].parameters() if p.requires_grad)
    print(f"\nEncoder parameters: {n_params}")

    cluster_optims = [
        torch.optim.Adam(
            cluster_models[c].parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        for c in range(num_clusters)
    ]

    cluster_scheds = [
        MultiStepLR(
            cluster_optims[c],
            milestones=[args.epochs // 2, args.epochs // 4 * 3],
            gamma=0.1,
        )
        for c in range(num_clusters)
    ]

    # ===========================================================
    # Step 5: Train
    # ===========================================================
    trainer = ClusterTrainer(
        cluster_models=cluster_models,
        cluster_optims=cluster_optims,
        cluster_scheds=cluster_scheds,
        cluster_loaders=cluster_loaders,
        patient_loaders=patient_loaders,
        cluster_assignments=cluster_assignments,
        groups=groups,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
