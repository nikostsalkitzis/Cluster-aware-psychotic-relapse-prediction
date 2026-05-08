"""
trainer.py
----------
Two-stage trainer for the cluster-shared-encoder + personal-ensemble pipeline.

Stage 1  (train_shared_encoder):
    One TransformerHeartPredictor per cluster, trained jointly on all
    cluster patients with inverse-frequency sample weighting.

Stage 2  (train_personal_heads):
    Encoder frozen.  One EnsembleLinear head per patient, trained on that
    patient's own windows only.

Validation (validate):
    For each patient: load their cluster encoder (frozen) + personal head,
    compute ensemble variance anomaly scores, compute AUROC / AUPRC.
"""

import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
from model import EnsembleLinear


# -------------------------------------------------------
# Factory: create one ensemble MLP head
# -------------------------------------------------------
def create_ensemble_mlp(args):
    m = nn.Sequential(
        EnsembleLinear(args.d_model, args.d_model, args.ensembles),
        nn.ReLU(),
        EnsembleLinear(args.d_model, args.d_model, args.ensembles),
        nn.ReLU(),
        EnsembleLinear(args.d_model, args.output_dim, args.ensembles),
    )
    m.to(args.device)
    return m


# -------------------------------------------------------
# Weighted MSE loss (used for shared encoder training)
# -------------------------------------------------------
def weighted_mse_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    preds, targets : (batch, output_dim)
    weights        : (batch,)  — per-sample inverse-frequency weights
    """
    per_sample = torch.mean((preds - targets) ** 2, dim=1)   # (batch,)
    return torch.mean(per_sample * weights)


# -------------------------------------------------------
# ClusterTrainer
# -------------------------------------------------------
class ClusterTrainer:
    """
    Manages training for the full cluster-aware pipeline.

    Parameters
    ----------
    cluster_models  : list[TransformerHeartPredictor]  — one per cluster
    cluster_optims  : list[Optimizer]                  — one per cluster
    cluster_scheds  : list[Scheduler]                  — one per cluster
    cluster_loaders : list[dict]  — each dict has keys "train", "val", "train_dist"
                                     indexed by cluster_id
    patient_loaders : list[dict]  — each dict has keys "train", "val", "train_dist"
                                     indexed by patient_id (0-based)
    cluster_assignments : dict  {patient_str: cluster_id}
    groups              : dict  {cluster_id: [patient_str]}
    args                : argparse.Namespace
    """

    def __init__(
        self,
        cluster_models,
        cluster_optims,
        cluster_scheds,
        cluster_loaders,
        patient_loaders,
        cluster_assignments,
        groups,
        args,
    ):
        self.cluster_models = cluster_models
        self.cluster_optims = cluster_optims
        self.cluster_scheds = cluster_scheds
        self.cluster_loaders = cluster_loaders
        self.patient_loaders = patient_loaders
        self.cluster_assignments = cluster_assignments   # {"P1": 0, "P3": 0, ...}
        self.groups = groups                             # {0: ["P1","P3"], 1: [...]}
        self.args = args
        self.num_clusters = len(cluster_models)
        self.num_patients = args.num_patients

        self.regression_loss = nn.MSELoss()

        # ---- Personal ensemble heads (one per patient, 0-indexed) ----
        self.personal_mlps = []
        self.personal_optims = []
        for _ in range(self.num_patients):
            head = create_ensemble_mlp(args)
            self.personal_mlps.append(head)
            self.personal_optims.append(
                torch.optim.Adam(head.parameters(), lr=args.head_learning_rate, weight_decay=1e-4)
            )

        # Best metric tracking per patient
        self.best_avgs   = [-np.inf] * self.num_patients
        self.best_aurocs = [-np.inf] * self.num_patients
        self.best_auprcs = [-np.inf] * self.num_patients

    # ==============================================================
    # STAGE 1 — Train shared encoder for one cluster, one epoch
    # ==============================================================
    def _train_encoder_epoch(self, cluster_id: int, epoch_metrics: dict) -> dict:
        """
        Train the shared encoder for cluster `cluster_id` on one epoch.
        Uses weighted MSE so every patient in the cluster contributes equally.
        """
        model = self.cluster_models[cluster_id]
        optim = self.cluster_optims[cluster_id]
        model.train()

        loader = self.cluster_loaders[cluster_id]["train"]
        desc = f"Encoder Cluster {cluster_id} (epoch)"

        for batch in tqdm(loader, desc=desc):
            if batch is None:
                continue

            x            = batch["data"].to(self.args.device)
            heart_labels = batch["target"].to(self.args.device)
            heart_labels = torch.squeeze(heart_labels, dim=-1)   # (B, 5)
            weights      = batch["sample_weight"].to(self.args.device)  # (B,)

            features, heart_preds = model(x)

            loss = weighted_mse_loss(heart_preds, heart_labels, weights)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()

            epoch_metrics["loss_total"] = epoch_metrics.get("loss_total", []) + [loss.item()]
            epoch_metrics["loss_heart"] = epoch_metrics.get("loss_heart", []) + [loss.item()]

        return epoch_metrics

    # ==============================================================
    # STAGE 2 — Train personal ensemble head for one patient
    # (encoder is FROZEN — no gradients through it)
    # ==============================================================
    def _resample_batch(self, indices, dataset):
        k = self.args.ensembles
        offsets = torch.zeros_like(indices)
        data_batch, target_batch = [], []
        for i in range(indices.size(0)):
            while (offsets[i] % k) == 0:
                offsets[i] = torch.randint(low=1, high=len(dataset), size=(1,))
            r_idx = (offsets[i] + indices[i]) % len(dataset)
            item = dataset[r_idx]
            data_batch.append(item["data"])
            target_batch.append(item["target"])
        return (
            torch.stack(data_batch).to(self.args.device),
            torch.stack(target_batch).to(self.args.device),
        )

    def _train_personal_head(self, patient_id: int):
        """
        Train personal ensemble head for patient `patient_id` (0-based).
        The cluster encoder for this patient is used in eval mode (frozen).
        """
        patient_str = f"P{patient_id + 1}"
        cluster_id  = self.cluster_assignments[patient_str]
        encoder     = self.cluster_models[cluster_id]

        k    = self.args.ensembles
        head = self.personal_mlps[patient_id]
        optim = self.personal_optims[patient_id]
        head.train()

        # Encoder is frozen — set eval and disable grad
        encoder.eval()

        loader  = self.patient_loaders[patient_id]["train"]
        dataset = loader.dataset
        desc    = f"Personal Head P{patient_id + 1}"

        for batch in tqdm(loader, desc=desc):
            if batch is None:
                continue

            x       = batch["data"].to(self.args.device)
            targets = batch["target"].to(self.args.device)
            targets = torch.squeeze(targets, dim=-1)          # (B, 5)

            # Extract features — no grad through encoder
            with torch.no_grad():
                features, _ = encoder(x)

            batched_features = features[None, :, :].repeat([k, 1, 1])
            batched_targets  = targets[None, :, :].repeat([k, 1, 1])
            ensemble_mask    = batch["idx"] % k

            resampled_x, r_targets = self._resample_batch(batch["idx"], dataset)
            r_targets = torch.squeeze(r_targets, dim=-1)
            with torch.no_grad():
                r_features, _ = encoder(resampled_x)

            batched_features[ensemble_mask] = r_features
            batched_targets[ensemble_mask]  = r_targets

            preds    = head(batched_features)
            mse_loss = torch.sum(torch.pow(preds - batched_targets, 2), dim=(0, 2))
            loss     = torch.mean(mse_loss)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5)
            optim.step()

    # ==============================================================
    # Compute personal train-distribution anomaly scores
    # ==============================================================
    def _get_train_dist(self, patient_id: int) -> list:
        patient_str = f"P{patient_id + 1}"
        cluster_id  = self.cluster_assignments[patient_str]
        encoder     = self.cluster_models[cluster_id]
        head        = self.personal_mlps[patient_id]
        k           = self.args.ensembles

        encoder.eval()
        head.eval()
        scores = []

        for batch in self.patient_loaders[patient_id]["train_dist"]:
            if batch is None:
                continue
            x = batch["data"].to(self.args.device)
            with torch.no_grad():
                features, _ = encoder(x)
                preds = head(features[None, :, :].repeat([k, 1, 1]))
                avg   = torch.mean(preds, 0)
                var   = torch.sum((preds - avg) ** 2, dim=2)
                score = torch.mean(torch.mean(var, 0)).item()
            scores.append(score)

        return scores

    # ==============================================================
    # Validation: compute anomaly scores + AUROC/AUPRC per patient
    # ==============================================================
    def _evaluate_patient(self, patient_id: int, train_dist: list):
        patient_str = f"P{patient_id + 1}"
        cluster_id  = self.cluster_assignments[patient_str]
        encoder     = self.cluster_models[cluster_id]
        head        = self.personal_mlps[patient_id]
        k           = self.args.ensembles

        encoder.eval()
        head.eval()

        _mean = float(np.mean(train_dist))
        _max  = float(np.max(train_dist))
        _min  = float(np.min(train_dist))
        denom = (_max - _min) if (_max - _min) != 0 else 1.0

        anomaly_scores, relapse_labels, user_ids = [], [], []

        for batch in self.patient_loaders[patient_id]["val"]:
            if batch is None:
                continue

            x = batch["data"].to(self.args.device).squeeze(0)
            with torch.no_grad():
                features, _ = encoder(x)
                batched     = features[None, :, :].repeat([k, 1, 1])
                preds       = head(batched)
                avg         = torch.mean(preds, 0)
                var         = torch.sum((preds - avg) ** 2, dim=2)
                mean_var    = torch.mean(torch.mean(var, 0)).item()

            score = (mean_var - _mean) / denom
            anomaly_scores.append(score)
            relapse_labels.append(batch["relapse_label"].item())
            user_ids.append(batch["user_id"].item())

        anomaly_scores = (np.array(anomaly_scores) > 0.0).astype(np.float64)
        relapse_labels = np.array(relapse_labels)
        user_ids       = np.array(user_ids)
        return anomaly_scores, relapse_labels, user_ids

    def _calc_metrics(self, anomaly_scores, relapse_labels):
        if len(np.unique(relapse_labels)) < 2:
            return None, None
        fpr, tpr, _        = sklearn.metrics.roc_curve(relapse_labels, anomaly_scores)
        prec, rec, _       = sklearn.metrics.precision_recall_curve(relapse_labels, anomaly_scores)
        auroc = sklearn.metrics.auc(fpr, tpr)
        auprc = sklearn.metrics.auc(rec, prec)
        return auroc, auprc

    # ==============================================================
    # Save best checkpoint for a patient
    # ==============================================================
    def _save_patient_checkpoint(self, patient_id: int, train_dist: list):
        patient_str = f"P{patient_id + 1}"
        cluster_id  = self.cluster_assignments[patient_str]
        encoder     = self.cluster_models[cluster_id]
        head        = self.personal_mlps[patient_id]

        pid_dir = os.path.join(self.args.save_path, str(patient_id + 1))
        os.makedirs(pid_dir, exist_ok=True)

        torch.save(encoder.state_dict(),
                   os.path.join(pid_dir, "best_encoder.pth"))
        torch.save(head.state_dict(),
                   os.path.join(pid_dir, "best_ensembles.pth"))

        dist_dict = {patient_id: train_dist}
        with open(os.path.join(pid_dir, "train_dist_anomaly_scores.pkl"), "wb") as f:
            pickle.dump(dist_dict, f)

    # ==============================================================
    # Full validation pass over all patients
    # ==============================================================
    def validate(self, epoch: int, epoch_metrics: dict, train_dists: dict):
        valid_aurocs, valid_auprcs = [], []

        for pid in range(self.num_patients):
            train_dist = train_dists.get(pid, [0.0])
            if not train_dist:
                train_dist = [0.0]

            scores, labels, uids = self._evaluate_patient(pid, train_dist)
            auroc, auprc = self._calc_metrics(scores, labels)

            if auroc is None:
                print(f"P{pid+1}: skipped (constant labels)")
                continue

            avg = (auroc + auprc) / 2
            if avg > self.best_avgs[pid]:
                self.best_avgs[pid]   = avg
                self.best_aurocs[pid] = auroc
                self.best_auprcs[pid] = auprc
                self._save_patient_checkpoint(pid, train_dist)

            valid_aurocs.append(self.best_aurocs[pid])
            valid_auprcs.append(self.best_auprcs[pid])

            print(
                f"P{pid+1}  AUROC={self.best_aurocs[pid]:.4f}  "
                f"AUPRC={self.best_auprcs[pid]:.4f}  "
                f"AVG={self.best_avgs[pid]:.4f}"
            )

        if valid_aurocs:
            total_auroc = float(np.mean(valid_aurocs))
            total_auprc = float(np.mean(valid_auprcs))
            total_avg   = (total_auroc + total_auprc) / 2
            train_loss  = float(np.mean(epoch_metrics.get("loss_total", [0])))
            print(
                f"\nTOTAL  AUROC={total_auroc:.4f}  AUPRC={total_auprc:.4f}  "
                f"AVG={total_avg:.4f}  TrainLoss={train_loss:.4f}\n"
            )

    # ==============================================================
    # Master training loop
    # ==============================================================
    def train(self):
        for epoch in range(self.args.epochs):
            print("*" * 60)
            print(f"Epoch {epoch + 1} / {self.args.epochs}")
            print("*" * 60)

            epoch_metrics = {}

            # --------------------------------------------------
            # Stage 1: train each shared encoder for one epoch
            # --------------------------------------------------
            for c_id in range(self.num_clusters):
                epoch_metrics = self._train_encoder_epoch(c_id, epoch_metrics)
                self.cluster_scheds[c_id].step()

            # --------------------------------------------------
            # Stage 2: train personal heads (encoder frozen)
            # --------------------------------------------------
            for pid in range(self.num_patients):
                self._train_personal_head(pid)

            # --------------------------------------------------
            # Compute personal train distributions
            # --------------------------------------------------
            train_dists = {}
            with torch.no_grad():
                for pid in range(self.num_patients):
                    train_dists[pid] = self._get_train_dist(pid)

            # --------------------------------------------------
            # Validate
            # --------------------------------------------------
            with torch.no_grad():
                self.validate(epoch, epoch_metrics, train_dists)
