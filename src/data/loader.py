"""Load per-flow JSON.gz records into numpy arrays / pandas DataFrames, with
label encoding and a confusion-matrix plotting helper."""

import os
import json
import gzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics


def encode_label(labels, class_label_pairs=None):
    """Encode string labels into integer class indices.

    Args:
        labels (iterable[str]): label strings, one per flow.
        class_label_pairs (dict[str, int] | None): existing class -> index
            mapping to reuse (keeps val/test indices consistent with training).
            If None, a new mapping is built from the sorted unique labels seen.

    Returns:
        tuple[np.ndarray, dict[str, int]]: integer label array and the
            class -> index mapping used.

    Raises:
        KeyError: a label is missing from ``class_label_pairs`` — a genuinely
            new/zero-day class must be handled separately, not encoded against
            the training label set.
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
    """One-hot encode integer label indices.

    Args:
        y_ (np.ndarray): 1-D integer class indices.
        n_classes (int | None): number of columns; if None, derived as ``max(y_) + 1``.

    Returns:
        np.ndarray: one-hot matrix of shape ``[len(y_), n_classes]``.
    """
    if n_classes is None:
        n_classes = int(max(y_)) + 1
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]


def read_json_gz(json_filename, feature_dict, max_rows=None):
    """Read one .json.gz file of per-flow JSON records, extracting the features
    listed in ``feature_dict``.

    ``feature_dict`` maps feature name -> -1 (take all sub-indices) or a list of
    specific indices to take. Must be provided explicitly — no hidden default.

    Rows are streamed (each parsed dict is released immediately) and collected
    into a single float32 array sized to the real max feature count, avoiding
    the previous (n, 2048) float64 preallocation.

    Returns:
        dataArray      : np.array [n_samples, n_features_selected] (float32)
        ids            : list of flow IDs, one per row
        feature_header : list of feature column names, in order
    """
    feature_header = []
    # Per-row float32 arrays (~0.2 GB at 387k rows) instead of Python-float
    # lists (~2 GB) — the difference between fitting and OOM on 8 GB.
    blocks = []
    ids = []
    skipped_lines = []

    with gzip.open(json_filename, "rb") as jj:
        line_no = 0
        while True:
            line_no += 1
            raw = jj.readline()
            if not raw:
                break
            try:
                flow = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                skipped_lines.append(line_no)
                continue

            ids.append(flow["id"])
            row = []
            # (max_rows cap checked after the row is parsed, below)
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
                        row.append(extracted[j])
                        col_name = f"{feature}_{j}"
                        if col_name not in feature_header:
                            feature_header.append(col_name)
                elif isinstance(extracted, str):
                    continue  # categorical/string fields skipped, as in original
                else:
                    row.append(extracted)
                    if feature not in feature_header:
                        feature_header.append(feature)
            blocks.append(np.asarray(row, dtype=np.float32))
            if max_rows is not None and len(blocks) >= max_rows:
                print(f"Stopped early at max_rows={max_rows} in {json_filename}.")
                break

    if skipped_lines:
        print(f"Skipped {len(skipped_lines)} unparseable line(s) in {json_filename}.")

    if not blocks:
        return np.zeros((0, 0), dtype=np.float32), [], []

    col_counter_final = max(b.shape[0] for b in blocks)
    data_array = np.zeros((len(blocks), col_counter_final), dtype=np.float32)
    for i, b in enumerate(blocks):
        data_array[i, : b.shape[0]] = b

    return data_array, ids, feature_header


def read_dataset(dataset_folder, feature_dict, annotation_file=None, class_label_pairs=None, max_rows=None):
    """Walk ``dataset_folder`` for .json.gz files, extract features via
    ``feature_dict``, and optionally attach labels from ``annotation_file``.

    Works for any dataset sharing this per-flow JSON.gz schema with a matching
    ``feature_dict``. Verify the schema first; a mismatched ``feature_dict``
    silently produces wrong/empty columns rather than an error.
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
            d, ids, f_names = read_json_gz(os.path.join(root, f), feature_dict, max_rows=max_rows)

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


def get_training_data(training_folder, annotation_file, feature_dict, max_rows=None):
    """Load training data as (Xtrain, y_train, class_label_pairs, ids).

    ``max_rows`` caps rows read (tracer-bullet runs; None = full file). Returns
    the raw float32 numpy array directly — the previous numpy -> pandas ->
    .values round trip doubled peak memory for no benefit.
    """
    print("\nLoading training set ...")
    feature_names, ids, X, y, clp = read_dataset(
        training_folder, feature_dict, annotation_file, class_label_pairs=None,
        max_rows=max_rows,
    )
    return X, y, clp, ids


def get_labeled_eval_data(eval_folder, annotation_file, feature_dict, class_label_pairs):
    """Load a labeled held-out evaluation set (e.g. NetML's 1_test-std_set),
    reusing the ``class_label_pairs`` learned from training so class indices
    line up. Use for local evaluation, not leaderboard submission.
    """
    print("\nLoading evaluation set ...")
    feature_names, ids, X, y, _ = read_dataset(
        eval_folder, feature_dict, annotation_file, class_label_pairs=class_label_pairs
    )
    df = pd.DataFrame(X, columns=feature_names)
    return df.values, y, ids


def plot_confusion_matrix(directory, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """Compute and save a confusion-matrix plot with TPR/FAR (binary) or
    F1/mAP (multi-class) in the title.

    Args:
        directory (str): output directory; the figure is written to ``<directory>/CM.png``.
        y_true (array-like): ground-truth integer labels.
        y_pred (array-like): predicted integer labels.
        classes (list[str]): display names, indexed by class index.
        normalize (bool): if True, plot row-normalized counts instead of raw.
        title (str | None): figure title; auto-generated when None.
        cmap (matplotlib colormap): cell coloring (default ``plt.cm.Blues``).

    Returns:
        tuple[matplotlib.axes.Axes, np.ndarray]: the plot axes and the
            (possibly normalized) confusion matrix.
    """
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