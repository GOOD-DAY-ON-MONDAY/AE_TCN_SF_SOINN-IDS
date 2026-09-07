"""Load per-flow JSON.gz records into numpy arrays, with label encoding."""

import gzip
import json
import os

import numpy as np


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
                    indices = (
                        range(len(extracted))
                        if feature_dict[feature] == -1
                        else feature_dict[feature]
                    )
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


def read_dataset(
    dataset_folder,
    feature_dict,
    annotation_file=None,
    class_label_pairs=None,
    max_rows=None,
):
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
            d, ids, f_names = read_json_gz(
                os.path.join(root, f), feature_dict, max_rows=max_rows
            )

            if len(f_names) > len(feature_names):
                feature_names = f_names

            data_array = (
                d if data_array is None else np.concatenate((data_array, d), axis=0)
            )
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
    _, ids, X, y, clp = read_dataset(
        training_folder,
        feature_dict,
        annotation_file,
        class_label_pairs=None,
        max_rows=max_rows,
    )
    return X, y, clp, ids



