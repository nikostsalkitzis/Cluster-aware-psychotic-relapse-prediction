"""
cluster_utils.py  —  Ward-linkage hierarchical clustering on patient profiles
==============================================================================
Patients are represented by their median physiological profile over training
data.  Hierarchical agglomerative clustering with Ward linkage and Euclidean
distance minimises within-cluster variance at each merge step.

Public entry point
------------------
    assignments, groups = build_clusters(
        features_path, num_patients, n_clusters, save_path
    )
"""

import os
import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_COLS = [
    "acc_norm", "gyr_norm", "heartRate_mean", "rRInterval_mean",
    "rRInterval_rmssd", "rRInterval_sdnn",
    "rRInterval_lombscargle_power_high", "steps",
]

PATIENT_DIAGNOSIS = {
    "P1": "Brief Psychotic Episode",
    "P2": "Schizoaffective Disorder",
    "P3": "Schizophrenia",
    "P4": "Bipolar I Disorder",
    "P5": "Schizophrenia",
    "P6": "Bipolar I Disorder",
    "P7": "Bipolar I Disorder",
    "P8": "Bipolar I Disorder",
}

DIAGNOSIS_COLOUR = {
    "Schizophrenia":            "#1F4E8C",
    "Schizoaffective Disorder": "#4C72B0",
    "Brief Psychotic Episode":  "#7EA6D4",
    "Bipolar I Disorder":       "#2E9E7A",
}

HULL_FILL = {
    "Schizophrenia Spectrum": "#B8D0EC",
    "Bipolar I Disorder":     "#B0E4D0",
}
HULL_EDGE = {
    "Schizophrenia Spectrum": "#1F4E8C",
    "Bipolar I Disorder":     "#2E9E7A",
}

LINKAGE_METHOD  = "ward"
DISTANCE_METRIC = "euclidean"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _broad_group(diagnosis):
    if diagnosis in ("Schizophrenia", "Schizoaffective Disorder", "Brief Psychotic Episode"):
        return "Schizophrenia Spectrum"
    return diagnosis


def _dominant_group(patient_list):
    groups = [_broad_group(PATIENT_DIAGNOSIS.get(p, "Unknown")) for p in patient_list]
    return max(set(groups), key=groups.count)


def _hull_fill(dominant):
    return HULL_FILL.get(dominant, "#D8D8D8")


def _hull_edge(dominant):
    return HULL_EDGE.get(dominant, "#888888")


def _expand_hull(verts, centroid, margin=1.5):
    directions = verts - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return verts + (directions / norms) * margin


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Compute median physiological profile per patient
# ─────────────────────────────────────────────────────────────────────────────
def compute_patient_profiles(features_path, num_patients):
    """
    For each patient, load all training-split feature CSVs and compute
    the median value across all timesteps for each sensor column.
    Returns a dict {patient_str: np.ndarray of shape (8,)}.
    """
    profiles = {}
    for pid in range(1, num_patients + 1):
        patient = f"P{pid}"
        patient_dir = os.path.join(features_path, patient)
        if not os.path.isdir(patient_dir):
            print(f"Warning: patient directory not found: {patient_dir}, skipping.")
            continue

        all_rows = []
        for subfolder in os.listdir(patient_dir):
            if not ("train" in subfolder and subfolder.endswith("train")):
                continue
            subfolder_dir = os.path.join(patient_dir, subfolder)
            for fname in os.listdir(subfolder_dir):
                if not fname.endswith("features_stretched_w_steps.csv"):
                    continue
                df = pd.read_csv(os.path.join(subfolder_dir, fname))
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                present = [c for c in SENSOR_COLS if c in df.columns]
                if present:
                    all_rows.append(df[present])

        if not all_rows:
            print(f"Warning: no training data found for {patient}, skipping.")
            continue

        combined = pd.concat(all_rows, ignore_index=True)
        profiles[patient] = combined[SENSOR_COLS].median().to_numpy()
        print(f"  {patient}: profile computed from {len(combined)} timesteps")

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Hierarchical agglomerative clustering (Ward + Euclidean)
# ─────────────────────────────────────────────────────────────────────────────
def cluster_patients(profiles, n_clusters, save_path):
    """
    Cluster patients by their standardised median physiological profiles
    using Ward linkage with Euclidean distance.  The optimal partition is
    selected by cutting the dendrogram at n_clusters.

    Returns
    -------
    assignments : dict {patient_str: cluster_id}
    groups      : dict {cluster_id: [patient_str, ...]}
    X_scaled    : np.ndarray (n_patients, n_features) — standardised profiles
    patients    : list of patient strings (sorted)
    """
    patients = sorted(profiles.keys())
    X = np.stack([profiles[p] for p in patients])

    # Z-score standardise so no single sensor dominates by scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ward linkage requires Euclidean distance
    condensed_D = pdist(X_scaled, metric=DISTANCE_METRIC)
    Z = linkage(condensed_D, method=LINKAGE_METHOD)

    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    assignments = {p: int(labels[i]) for i, p in enumerate(patients)}

    groups = {}
    for p, c in assignments.items():
        groups.setdefault(c, []).append(p)

    print("\n" + "=" * 50)
    print(f"Clustering result  ({n_clusters} clusters)"
          f"  [method={LINKAGE_METHOD}, metric={DISTANCE_METRIC}]")
    print("=" * 50)
    for c_id, members in sorted(groups.items()):
        dominant = _dominant_group(members)
        print(f"  Cluster {c_id}: {members}  → dominant group: {dominant}")
    print("=" * 50 + "\n")

    os.makedirs(save_path, exist_ok=True)
    apath = os.path.join(save_path, "cluster_assignments.pkl")
    gpath = os.path.join(save_path, "cluster_groups.pkl")
    with open(apath, "wb") as f:
        pickle.dump(assignments, f)
    with open(gpath, "wb") as f:
        pickle.dump(groups, f)
    print(f"Saved cluster_assignments.pkl -> {apath}")
    print(f"Saved cluster_groups.pkl      -> {gpath}")

    return assignments, groups, X_scaled, patients


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — PCA + t-SNE overview (cluster colours)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_hulls_filled_dark(ax, X_emb, patients, assignments, cluster_colours, alpha=0.15):
    from scipy.spatial import ConvexHull
    for cid in sorted(set(assignments.values())):
        pts = np.array([X_emb[i] for i, p in enumerate(patients)
                        if assignments[p] == cid])
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            poly = plt.Polygon(
                pts[hull.vertices], closed=True,
                facecolor=cluster_colours[cid], edgecolor=cluster_colours[cid],
                alpha=alpha, linewidth=1.2, zorder=1,
            )
            ax.add_patch(poly)
        except Exception:
            pass


def plot_cluster_embeddings(X_scaled, patients, assignments, n_clusters,
                            save_path, tsne_perplexity=5, random_state=42):
    """
    Side-by-side PCA and t-SNE scatter plots coloured by cluster assignment.
    Saved to <save_path>/patient_cluster_plot.png.
    """
    n = len(patients)
    perplexity = min(tsne_perplexity, max(1, n - 1))

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_scaled)

    cmap = plt.get_cmap("tab10")
    cluster_colours = {c: cmap(c / max(n_clusters - 1, 1)) for c in range(n_clusters)}
    BG, GRID, TEXT = "#0d1b2a", "#1e3050", "#cfe8ff"

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor=BG)
    panel_data = [
        (axes[0], X_pca,
         f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
         f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"),
        (axes[1], X_tsne, "t-SNE Dimension 1", "t-SNE Dimension 2"),
    ]

    for ax, X_emb, xlabel, ylabel in panel_data:
        ax.set_facecolor(BG)
        ax.set_xlabel(xlabel, color=TEXT, fontsize=16)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=16)
        ax.tick_params(colors=TEXT, labelsize=13)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
        _draw_hulls_filled_dark(ax, X_emb, patients, assignments, cluster_colours)
        for i, pat in enumerate(patients):
            cid = assignments[pat]
            ax.scatter(X_emb[i, 0], X_emb[i, 1],
                       color=cluster_colours[cid], marker="o",
                       s=220, edgecolors="white", linewidths=0.8, zorder=5)
            ax.annotate(pat, (X_emb[i, 0], X_emb[i, 1]),
                        textcoords="offset points", xytext=(7, 5),
                        fontsize=14, color=TEXT, fontweight="bold", zorder=6)

    legend_handles = [
        mpatches.Patch(facecolor=cluster_colours[c], edgecolor="white",
                       label=f"Cluster {c}")
        for c in range(n_clusters)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=n_clusters,
               frameon=False, fontsize=14, labelcolor=TEXT, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, 0.98,
             f"Clustering: {LINKAGE_METHOD.capitalize()} linkage  ·  "
             f"{DISTANCE_METRIC.capitalize()} distance",
             ha="center", va="top", fontsize=13, color=TEXT, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plot_path = os.path.join(save_path, "patient_cluster_plot.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved cluster plot          -> {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — t-SNE coloured by clinical diagnosis (paper Figure 3)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_diagnosis_hulls(ax, X_emb, patients, assignments, groups, cluster_dominant):
    from scipy.spatial import ConvexHull
    centroids = {}
    for cid, members in sorted(groups.items()):
        pts = np.array([X_emb[i] for i, p in enumerate(patients)
                        if assignments[p] == cid])
        dominant  = cluster_dominant[cid]
        fill_col  = _hull_fill(dominant)
        edge_col  = _hull_edge(dominant)
        centroids[cid] = pts.mean(axis=0)
        if len(pts) < 3:
            cx, cy = centroids[cid]
            circle = plt.Circle(
                (cx, cy), radius=2.5,
                facecolor=fill_col, edgecolor=edge_col,
                linewidth=2.0, linestyle="--", alpha=0.45, zorder=1,
            )
            ax.add_patch(circle)
            continue
        try:
            hull = ConvexHull(pts)
            expanded = _expand_hull(pts[hull.vertices], centroids[cid], margin=1.5)
            poly = plt.Polygon(
                expanded, closed=True,
                facecolor=fill_col, edgecolor=edge_col,
                alpha=0.38, linewidth=2.2, linestyle="--", zorder=1,
            )
            ax.add_patch(poly)
        except Exception:
            pass
    return centroids


def _annotate_clusters_inside(ax, centroids, cluster_dominant, ax_xlim, ax_ylim):
    display_name = {
        "Schizophrenia Spectrum": "Schizophrenia\nSpectrum",
        "Bipolar I Disorder":     "Bipolar I\nDisorder",
    }
    x_min, x_max = ax_xlim
    y_min, y_max = ax_ylim
    x_range, y_range = x_max - x_min, y_max - y_min
    cids = sorted(centroids.keys())
    for cid in cids:
        centroid  = centroids[cid]
        dominant  = cluster_dominant[cid]
        edge_col  = _hull_edge(dominant)
        label     = display_name.get(dominant, dominant)
        full_label = f"$\\mathcal{{C}}_{{{cid}}}$\n{label}"
        others    = np.array([centroids[o] for o in cids if o != cid])
        other_mean = others.mean(axis=0) if len(others) else centroid
        direction = centroid - other_mean
        norm      = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        dx = direction[0] * x_range * 0.15 * 1.6
        dy = direction[1] * y_range * 0.15 * 1.6
        tx = np.clip(centroid[0] + dx, x_min + x_range * 0.18, x_max - x_range * 0.18)
        ty = np.clip(centroid[1] + dy, y_min + y_range * 0.15, y_max - y_range * 0.15)
        ax.annotate(
            full_label,
            xy=centroid, xytext=(tx, ty),
            fontsize=30, fontweight="bold", color=edge_col,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            multialignment="center",
            bbox=dict(boxstyle="round,pad=0.65", facecolor="white",
                      edgecolor=edge_col, linewidth=2.0, alpha=0.95),
            arrowprops=dict(arrowstyle="-|>", color=edge_col,
                            lw=2.2, connectionstyle="arc3,rad=0.15"),
            zorder=8,
        )


def _draw_legend_inside(ax, patients, groups, cluster_dominant):
    present_diagnoses = sorted(
        set(PATIENT_DIAGNOSIS[p] for p in patients if p in PATIENT_DIAGNOSIS),
        key=lambda d: list(DIAGNOSIS_COLOUR.keys()).index(d),
    )
    diag_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=DIAGNOSIS_COLOUR[d], markeredgecolor="white",
               markeredgewidth=0.8, markersize=22, label=d)
        for d in present_diagnoses
    ]
    display_name = {
        "Schizophrenia Spectrum": "Schizophrenia Spectrum",
        "Bipolar I Disorder":     "Bipolar I Disorder",
    }
    cluster_handles = [
        mpatches.Patch(
            facecolor=_hull_fill(cluster_dominant[cid]),
            edgecolor=_hull_edge(cluster_dominant[cid]),
            linewidth=1.8, linestyle="--", alpha=0.80,
            label=f"$\\mathcal{{C}}_{{{cid}}}$: "
                  f"{display_name.get(cluster_dominant[cid], cluster_dominant[cid])}",
        )
        for cid in sorted(groups.keys())
    ]
    blank          = mpatches.Patch(visible=False, label="")
    header_diag    = mpatches.Patch(visible=False, label="Diagnosis")
    header_cluster = mpatches.Patch(visible=False, label="Cluster")
    all_handles    = [header_diag] + diag_handles + [blank, header_cluster] + cluster_handles

    leg = ax.legend(
        handles=all_handles,
        loc="lower left", bbox_to_anchor=(0.01, 0.01),
        bbox_transform=ax.transAxes, borderaxespad=0,
        framealpha=0.97, edgecolor="#cccccc", fancybox=True,
        fontsize=24, handlelength=1.2, handleheight=0.9,
        borderpad=0.6, labelspacing=0.35, handletextpad=0.5,
    )
    leg.get_frame().set_linewidth(1.6)
    for text in leg.get_texts():
        if text.get_text().strip() in {"Diagnosis", "Cluster"}:
            text.set_fontweight("bold")
            text.set_fontsize(28)
            text.set_color("#111111")


def plot_tsne_diagnosis(X_scaled, patients, assignments, n_clusters,
                        save_path, tsne_perplexity=5, random_state=42):
    """
    t-SNE plot with patients coloured by clinical diagnosis and convex hulls
    drawn per cluster.  Reproduces Figure 3 of the paper.
    Saved to <save_path>/patient_tsne_diagnosis.png.
    """
    n = len(patients)
    perplexity = min(tsne_perplexity, max(1, n - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_scaled)

    groups = {}
    for p, c in assignments.items():
        groups.setdefault(c, []).append(p)
    cluster_dominant = {c: _dominant_group(members) for c, members in groups.items()}

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 32,
        "axes.titlesize": 36, "axes.labelsize": 34,
        "xtick.labelsize": 30, "ytick.labelsize": 30,
        "text.usetex": False,
    })

    fig, ax = plt.subplots(figsize=(22, 17), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.09)
    ax.set_facecolor("white")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=34, color="#444444")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=34, color="#444444")
    ax.tick_params(labelsize=30, colors="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
        spine.set_linewidth(0.8)
    ax.grid(color="#f2f2f2", linewidth=0.8, linestyle="--", alpha=1.0)
    ax.set_title(
        f"Clustering: {LINKAGE_METHOD.capitalize()} linkage  ·  "
        f"{DISTANCE_METRIC.capitalize()} distance",
        fontsize=26, color="#666666", loc="right", pad=8, style="italic",
    )

    hull_centroids = _draw_diagnosis_hulls(
        ax, X_tsne, patients, assignments, groups, cluster_dominant
    )
    for i, pat in enumerate(patients):
        diag   = PATIENT_DIAGNOSIS.get(pat, "Unknown")
        colour = DIAGNOSIS_COLOUR.get(diag, "#999999")
        ax.scatter(X_tsne[i, 0], X_tsne[i, 1], color=colour, marker="o",
                   s=750, edgecolors="white", linewidths=2.8, zorder=5)
        ax.annotate(pat, (X_tsne[i, 0], X_tsne[i, 1]),
                    textcoords="offset points", xytext=(16, 9),
                    fontsize=30, color="#1a1a1a", fontweight="bold", zorder=6)

    ax.autoscale_view()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(x_min - (x_max - x_min) * 0.18, x_max + (x_max - x_min) * 0.18)
    ax.set_ylim(y_min - (y_max - y_min) * 0.18, y_max + (y_max - y_min) * 0.18)

    _annotate_clusters_inside(
        ax, hull_centroids, cluster_dominant,
        ax_xlim=ax.get_xlim(), ax_ylim=ax.get_ylim(),
    )
    _draw_legend_inside(ax, patients, groups, cluster_dominant)

    plot_path = os.path.join(save_path, "patient_tsne_diagnosis.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved t-SNE diagnosis plot  -> {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — called by train.py
# ─────────────────────────────────────────────────────────────────────────────
def build_clusters(features_path, num_patients, n_clusters, save_path,
                   tsne_perplexity=5):
    """
    Full clustering pipeline:
      1. Compute median physiological profiles per patient
      2. Cluster with Ward linkage + Euclidean distance
      3. Save assignments and generate embedding plots

    Returns
    -------
    assignments : dict {patient_str: cluster_id}
    groups      : dict {cluster_id: [patient_str, ...]}
    """
    print("=" * 50)
    print("Step 0: Computing patient profiles ...")
    print("=" * 50)
    profiles = compute_patient_profiles(features_path, num_patients)

    if len(profiles) < n_clusters:
        raise ValueError(
            f"Only {len(profiles)} patients found but n_clusters={n_clusters}."
        )

    print(f"\nClustering patients  "
          f"[linkage={LINKAGE_METHOD}, metric={DISTANCE_METRIC}] ...")
    assignments, groups, X_scaled, patients = cluster_patients(
        profiles, n_clusters, save_path
    )

    print("Generating embedding plots ...")
    plot_cluster_embeddings(X_scaled, patients, assignments, n_clusters,
                            save_path, tsne_perplexity=tsne_perplexity)
    plot_tsne_diagnosis(X_scaled, patients, assignments, n_clusters,
                        save_path, tsne_perplexity=tsne_perplexity)

    return assignments, groups


# ─────────────────────────────────────────────────────────────────────────────
# CLI — run clustering standalone without training
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Cluster patients by median sensor profile using "
            f"{LINKAGE_METHOD} linkage and {DISTANCE_METRIC} distance."
        )
    )
    parser.add_argument("--features_path",   type=str, required=True,
                        help="Path to extracted feature CSVs (track_2_new_features/)")
    parser.add_argument("--num_patients",    type=int, default=8)
    parser.add_argument("--n_clusters",      type=int, default=2)
    parser.add_argument("--save_path",       type=str, default="checkpoints_clustered")
    parser.add_argument("--tsne_perplexity", type=int, default=5)
    args = parser.parse_args()

    build_clusters(
        features_path   = args.features_path,
        num_patients    = args.num_patients,
        n_clusters      = args.n_clusters,
        save_path       = args.save_path,
        tsne_perplexity = args.tsne_perplexity,
    )
