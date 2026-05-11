import os
from typing import Optional, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# -------------------------------------------------------
# PatientDataset
# -------------------------------------------------------
class PatientDataset(Dataset):
    def __init__(
        self,
        features_path,
        dataset_path,
        patient: Optional[str] = None,
        mode: str = "train",
        scaler=None,
        window_size: int = 48,
        stride: int = 12,
    ):
        self.features_path = features_path
        self.dataset_path = dataset_path
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        self.columns_to_scale = [
            "acc_norm",
            "gyr_norm",
            "heartRate_mean",
            "rRInterval_mean",
            "rRInterval_rmssd",
            "rRInterval_sdnn",
            "rRInterval_lombscargle_power_high",
            "steps",
        ]
        self.data_columns = self.columns_to_scale

        self.target_columns = [
            "heartRate_mean",
            "rRInterval_mean",
            "rRInterval_rmssd",
            "rRInterval_sdnn",
            "rRInterval_lombscargle_power_high",
        ]

        self.data = []
        all_data = pd.DataFrame()

        if patient is None:
            for patient in sorted(os.listdir(features_path)):
                if patient == ".DS_Store":
                    continue
                all_data = self.create_data(patient, mode, all_data)
        else:
            all_data = self.create_data(patient, mode, all_data)

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_data[self.columns_to_scale].dropna().to_numpy())
        else:
            self.scaler = scaler

        print(f"Created dataset `{mode}` for Patient {patient} of size: {len(self)}")

    def create_data(self, patient: str, mode: str, all_data: pd.DataFrame):
        patient_dir = os.path.join(self.features_path, patient)
        for subfolder in os.listdir(patient_dir):
            if (
                ("train" in mode and "train" in subfolder and subfolder.endswith("train"))
                or (mode == "val" and "val" in subfolder and subfolder.endswith("val"))
                or (mode == "test" and "test" in subfolder)
            ):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                for file in os.listdir(subfolder_dir):
                    if file.endswith("features_stretched_w_steps.csv"):
                        file_path = os.path.join(subfolder_dir, file)
                        df = pd.read_csv(file_path)
                        df = df.replace([np.inf, -np.inf], np.nan)
                        df = df.dropna()

                        missing_targets = [c for c in self.target_columns if c not in df.columns]
                        if missing_targets:
                            print(f"Warning: Missing target columns {missing_targets} in {file_path}, skipping.")
                            continue

                        all_data = pd.concat([all_data, df])
                        day_indices = df["day"].unique()

                        relapse_df = None
                        if "train" not in mode:
                            relapse_data_path = os.path.join(
                                self.features_path, patient, subfolder, "relapse_stretched.csv"
                            )
                            relapse_df = pd.read_csv(relapse_data_path)

                        for day_index in day_indices:
                            day_data = df[df["day"] == day_index].copy()

                            relapse_label = 0
                            if "train" not in mode and relapse_df is not None:
                                try:
                                    relapse_label = relapse_df[
                                        relapse_df["day"] == day_index
                                    ]["relapse"].values[0]
                                except Exception:
                                    relapse_label = 0

                            if len(day_data) < self.window_size + 1:
                                continue

                            if mode == "train":
                                for start_idx in range(0, len(day_data) - self.window_size, self.stride):
                                    input_end = start_idx + self.window_size
                                    target_idx = input_end
                                    if target_idx >= len(day_data):
                                        break
                                    sequence = day_data.iloc[start_idx:input_end][self.data_columns].copy().to_numpy()
                                    target = day_data.iloc[target_idx:target_idx + 1][self.target_columns].copy().to_numpy()
                                    self.data.append((sequence, target, int(patient[1:]), relapse_label))
                            else:
                                self.data.append((day_data, None, int(patient[1:]), relapse_label))
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        day_data, target, patient_id, relapse_label = self.data[idx]

        if self.mode == "train":
            sequence = self.scaler.transform(day_data)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(1, 0)
            target_tensor = torch.tensor(target, dtype=torch.float32).permute(1, 0)
        else:
            sequences, targets = [], []
            if len(day_data) < self.window_size + 1:
                return None
            step = max(1, self.window_size // 3)
            for start_idx in range(0, len(day_data) - self.window_size, step):
                input_end = start_idx + self.window_size
                target_idx = input_end
                if target_idx >= len(day_data):
                    break
                seq = day_data.iloc[start_idx:input_end][self.data_columns].copy().to_numpy()
                seq = self.scaler.transform(seq)
                sequences.append(seq)
                tar = day_data.iloc[target_idx][self.target_columns].copy().to_numpy()
                targets.append(tar)

            sequence = np.stack(sequences)
            target = np.stack(targets)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0, 2, 1)
            target_tensor = torch.tensor(target, dtype=torch.float32)

        return {
            "data": sequence_tensor,
            "target": target_tensor,
            "user_id": torch.tensor(patient_id, dtype=torch.long) - 1,
            "relapse_label": torch.tensor(relapse_label, dtype=torch.long),
            "idx": idx,
        }


# -------------------------------------------------------
# ClusterDataset — used for shared encoder training (Stage 1)
# Loads ALL patients in a cluster jointly.
# Each window is tagged with:
#   - patient_id  (for personal head routing)
#   - sample_weight (inverse-frequency so every patient contributes equally)
# -------------------------------------------------------
class ClusterDataset(Dataset):
    def __init__(
        self,
        features_path: str,
        dataset_path: str,
        patients: List[str],          # e.g. ["P1", "P3", "P6"]
        mode: str = "train",
        scaler=None,
        window_size: int = 48,
        stride: int = 12,
    ):
        self.features_path = features_path
        self.dataset_path = dataset_path
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.patients = patients

        self.columns_to_scale = [
            "acc_norm",
            "gyr_norm",
            "heartRate_mean",
            "rRInterval_mean",
            "rRInterval_rmssd",
            "rRInterval_sdnn",
            "rRInterval_lombscargle_power_high",
            "steps",
        ]
        self.data_columns = self.columns_to_scale

        self.target_columns = [
            "heartRate_mean",
            "rRInterval_mean",
            "rRInterval_rmssd",
            "rRInterval_sdnn",
            "rRInterval_lombscargle_power_high",
        ]

        # self.data entries: (sequence_np, target_np, patient_id_int, relapse_label)
        self.data = []
        all_data = pd.DataFrame()

        # Count windows per patient so we can compute sample weights
        self.patient_window_counts = {}

        for patient in patients:
            before = len(self.data)
            all_data = self._create_data(patient, mode, all_data)
            after = len(self.data)
            self.patient_window_counts[patient] = after - before

        # ---- Fit or reuse scaler on the FULL cluster training data ----
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_data[self.columns_to_scale].dropna().to_numpy())
        else:
            self.scaler = scaler

        # ---- Compute per-window inverse-frequency sample weights ----
        # Every patient contributes equal total gradient mass regardless of window count.
        total_windows = len(self.data)
        num_patients = len(patients)
        # weight_i = (total / num_patients) / count_i
        # => patients with fewer windows get higher weight
        self.sample_weights = np.ones(total_windows, dtype=np.float32)
        cursor = 0
        for patient in patients:
            count = self.patient_window_counts[patient]
            if count > 0:
                weight = (total_windows / num_patients) / count
            else:
                weight = 1.0
            self.sample_weights[cursor: cursor + count] = weight
            cursor += count

        print(
            f"ClusterDataset `{mode}` | patients={patients} | "
            f"total windows={total_windows} | "
            f"per-patient counts={self.patient_window_counts}"
        )

    # ------------------------------------------------------------------
    def _create_data(self, patient: str, mode: str, all_data: pd.DataFrame):
        patient_dir = os.path.join(self.features_path, patient)
        if not os.path.isdir(patient_dir):
            print(f"Warning: patient dir not found: {patient_dir}")
            return all_data

        for subfolder in os.listdir(patient_dir):
            if (
                ("train" in mode and "train" in subfolder and subfolder.endswith("train"))
                or (mode == "val" and "val" in subfolder and subfolder.endswith("val"))
                or (mode == "test" and "test" in subfolder)
            ):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                for file in os.listdir(subfolder_dir):
                    if not file.endswith("features_stretched_w_steps.csv"):
                        continue
                    file_path = os.path.join(subfolder_dir, file)
                    df = pd.read_csv(file_path)
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()

                    missing = [c for c in self.target_columns if c not in df.columns]
                    if missing:
                        print(f"Warning: Missing columns {missing} in {file_path}, skipping.")
                        continue

                    all_data = pd.concat([all_data, df])
                    day_indices = df["day"].unique()

                    relapse_df = None
                    if "train" not in mode:
                        relapse_path = os.path.join(
                            self.features_path, patient, subfolder, "relapse_stretched.csv"
                        )
                        relapse_df = pd.read_csv(relapse_path)

                    for day_index in day_indices:
                        day_data = df[df["day"] == day_index].copy()
                        if len(day_data) < self.window_size + 1:
                            continue

                        relapse_label = 0
                        if "train" not in mode and relapse_df is not None:
                            try:
                                relapse_label = relapse_df[
                                    relapse_df["day"] == day_index
                                ]["relapse"].values[0]
                            except Exception:
                                relapse_label = 0

                        if mode == "train":
                            for start_idx in range(0, len(day_data) - self.window_size, self.stride):
                                input_end = start_idx + self.window_size
                                target_idx = input_end
                                if target_idx >= len(day_data):
                                    break
                                sequence = (
                                    day_data.iloc[start_idx:input_end][self.data_columns]
                                    .copy().to_numpy()
                                )
                                target = (
                                    day_data.iloc[target_idx:target_idx + 1][self.target_columns]
                                    .copy().to_numpy()
                                )
                                self.data.append(
                                    (sequence, target, int(patient[1:]), relapse_label)
                                )
                        else:
                            self.data.append(
                                (day_data, None, int(patient[1:]), relapse_label)
                            )
        return all_data

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        day_data, target, patient_id, relapse_label = self.data[idx]
        weight = float(self.sample_weights[idx])

        if self.mode == "train":
            sequence = self.scaler.transform(day_data)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(1, 0)
            target_tensor = torch.tensor(target, dtype=torch.float32).permute(1, 0)
        else:
            sequences, targets = [], []
            if len(day_data) < self.window_size + 1:
                return None
            step = max(1, self.window_size // 3)
            for start_idx in range(0, len(day_data) - self.window_size, step):
                input_end = start_idx + self.window_size
                target_idx = input_end
                if target_idx >= len(day_data):
                    break
                seq = day_data.iloc[start_idx:input_end][self.data_columns].copy().to_numpy()
                seq = self.scaler.transform(seq)
                sequences.append(seq)
                tar = day_data.iloc[target_idx][self.target_columns].copy().to_numpy()
                targets.append(tar)

            sequence = np.stack(sequences)
            target = np.stack(targets)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0, 2, 1)
            target_tensor = torch.tensor(target, dtype=torch.float32)

        return {
            "data": sequence_tensor,
            "target": target_tensor,
            "user_id": torch.tensor(patient_id, dtype=torch.long) - 1,
            "relapse_label": torch.tensor(relapse_label, dtype=torch.long),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }
