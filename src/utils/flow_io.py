"""
Loads per-flow JSON.gz flow records into
numpy arrays / pandas DataFrames, with label encoding and a
confusion-matrix plotting helper.
"""

import os
import json
import gzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics


def encode_label(labels, class_label_pairs=None):
    """
    Encode string labels into integer class indices.

    If class_label_pairs is None, builds a new mapping from the
    sorted unique labels seen (use this on the training set).
    If class_label_pairs is provided, reuses it (use this on val/
    test/zero-day-holdout sets so class indices stay consistent
    with training).
    """
    label_list = []

    if class_label_pairs is None:
        unique = sorted(set(labels))
        class_label_pairs = {ul: i for i, ul in enumerate(unique)}

    for label in labels:
        if label not in class_label_pairs:
            raise KeyError(
                f"Label '{label}' not found in class_label_pairs. "
                f"If this is a genuinely new/zero-day class, handle it "
                f"separately rather than encoding it against the "
                f"training label set."
            )
        label_list.append(class_label_pairs[label])

    label_array = np.asarray(label_list).reshape((-1,))
    return label_array, class_label_pairs


def one_hot(y_, n_classes=None):
    
    """One-hot encode integer label indices."""
    
    if n_classes is None:
        n_classes = int(max(y_)) + 1
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]


def read_json_gz(json_filename, feature_dict):
    
    """
    Read one .json.gz file of per-flow JSON records and extract the
    features listed in feature_dict.

    feature_dict: dict mapping feature_name -> -1 (take all
        sub-indices) or a list of specific indices to take.
        Must be provided explicitly — no hidden default file.

    Returns:
        dataArray      : np.array [n_samples, n_features_selected]
        ids            : list of flow IDs, one per row
        feature_header : list of feature column names, in order
    """
    feature_header = []
    data = []
    skipped_lines = []

    with gzip.open(json_filename, "rb") as jj:
        line_no = 0
        while True:
            line_no += 1
            raw = jj.readline()
            if not raw:
                break
            try:
                sample = json.loads(raw.decode("utf-8"))
                data.append(sample)
            except (UnicodeDecodeError, json.JSONDecodeError):
                skipped_lines.append(line_no)

    if skipped_lines:
        print(f"Skipped {len(skipped_lines)} unparseable line(s) in {json_filename}.")

    if not data:
        return np.zeros((0, 0)), [], []

    data_array = np.zeros((len(data), 2048))
    ids = []
    col_counter_final = 0

    for i, flow in enumerate(data):
        ids.append(flow["id"])
        col_counter = 0
        for feature in sorted(feature_dict.keys()):
            if feature not in flow:
                continue
            extracted = flow[feature]

            if isinstance(extracted, list):
                if len(extracted) == 0:
                    continue
                if isinstance(extracted[0], dict):
                    # e.g. SPLT / byte_dist stored as dict — not handled here
                    continue
                indices = range(len(extracted)) if feature_dict[feature] == -1 else feature_dict[feature]
                for j in indices:
                    data_array[i, col_counter] = extracted[j]
                    col_name = f"{feature}_{j}"
                    if col_name not in feature_header:
                        feature_header.append(col_name)
                    col_counter += 1
            elif isinstance(extracted, str):
                continue  # categorical/string fields skipped, as in original
            else:
                data_array[i, col_counter] = extracted
                if feature not in feature_header:
                    feature_header.append(feature)
                col_counter += 1

        col_counter_final = max(col_counter_final, col_counter)

    return data_array[:, :col_counter_final], ids, feature_header


def read_dataset(dataset_folder, feature_dict, annotation_file=None, class_label_pairs=None):
    """
    Walk dataset_folder for .json.gz files, extract features via
    feature_dict, and optionally attach labels from annotation_file.

    Works for any dataset that stores flows in this per-flow JSON.gz
    format with a matching feature_dict — including CICIDS2017, IF
    it shares this schema in the source repo. Verify this before
    relying on it; a mismatched feature_dict will silently produce
    wrong/empty columns rather than an error.
    """
    labels = []
    data_array = None
    feature_names = []
    all_ids = []

    for root, _, files in os.walk(dataset_folder):
        for f in files:
            if not f.endswith(".json.gz"):
                continue
            print(f"Reading {f}")
            d, ids, f_names = read_json_gz(os.path.join(root, f), feature_dict)

            if len(f_names) > len(feature_names):
                feature_names = f_names

            data_array = d if data_array is None else np.concatenate((data_array, d), axis=0)
            all_ids.extend(ids)

            if annotation_file is not None:
                with gzip.open(annotation_file, "rb") as an:
                    anno = json.loads(an.read().decode("utf-8"))
                for flow_id in ids:
                    labels.append(anno[str(flow_id)])

    if annotation_file is not None:
        label_array, class_label_pairs = encode_label(labels, class_label_pairs)
        return feature_names, all_ids, data_array, label_array, class_label_pairs

    return feature_names, all_ids, data_array, None, class_label_pairs


def get_training_data(training_folder, annotation_file, feature_dict):
    """Load training data as (Xtrain, y_train, class_label_pairs, ids)."""
    print("\nLoading training set ...")
    feature_names, ids, X, y, clp = read_dataset(
        training_folder, feature_dict, annotation_file, class_label_pairs=None
    )
    df = pd.DataFrame(X, columns=feature_names)
    return df.values, y, clp, ids


def get_labeled_eval_data(eval_folder, annotation_file, feature_dict, class_label_pairs):
    """
    Load a labeled held-out evaluation set (e.g. NetML's
    1_test-std_set), reusing the class_label_pairs learned from
    training so class indices line up.

    Use this instead of the competition's unlabeled submission-set
    loader — you're evaluating locally, not submitting to a
    leaderboard.
    """
    print("\nLoading evaluation set ...")
    feature_names, ids, X, y, _ = read_dataset(
        eval_folder, feature_dict, annotation_file, class_label_pairs=class_label_pairs
    )
    df = pd.DataFrame(X, columns=feature_names)
    return df.values, y, ids


def plot_confusion_matrix(directory, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """Compute and save a confusion matrix plot. Unchanged in behavior from the original."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if n_classes == 2:
        detection_rate = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        false_alarm_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        print(f"TPR: \t\t\t{detection_rate:.5f}")
        print(f"FAR: \t\t\t{false_alarm_rate:.5f}")
        if not title:
            label = "Normalized confusion matrix" if normalize else "Confusion matrix, without normalization"
            title = f"{label}\nTPR:{detection_rate:.5f} - FAR:{false_alarm_rate:.5f}"
    else:
        f1 = metrics.f1_score(y_true, y_pred, average="weighted")
        y_true_oh = one_hot(y_true, n_classes)
        y_pred_oh = one_hot(y_pred, n_classes)
        mAP = np.mean([
            metrics.average_precision_score(y_true_oh[:, c], y_pred_oh[:, c], average="weighted")
            for c in range(n_classes)
        ])
        print(f"F1: \t\t\t{f1:.5f}")
        print(f"mAP: \t\t\t{mAP:.5f}")
        if not title:
            label = "Normalized confusion matrix" if normalize else "Confusion matrix, without normalization"
            title = f"{label}\nF1:{f1:.5f} - mAP:{mAP:.5f}"

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm = cm_norm

    fig, ax = plt.subplots()
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fnt = 16 if n_classes < 4 else (10 if n_classes < 8 else max(4, 16 - n_classes))
    fmt = ".2f" if normalize else "d"
    thresh = np.sum(cm, axis=1) * 0.66

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] != 0:
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=fnt,
                    color="white" if cm[i, j] > thresh[i] else "black",
                )

    fig.tight_layout()
    out_path = os.path.join(directory, "CM.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Confusion matrix saved to {out_path}")

    return ax, cm