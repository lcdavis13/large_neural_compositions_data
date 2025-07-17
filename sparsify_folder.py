import os
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from typing import List

# === CONFIGURATION ===
rootdir = "synth/_data/256/"  # <-- Set your root directory here
overwrite = False  # Set to True to force re-sparsification even if output files exist

# File naming patterns
x_dense_prefix = "x0_"
x_sparse_prefix = "x0-sparse_"
mask_prefix = "_binary_"
ids_prefix = "ids-sparse_"
y_dense_suffix_pattern = re.compile(r"(.+)_y_(.+)\.csv$")
y_sparse_template = "{}_y-sparse_{}.csv"


# === DATACLASS FOR SPARSIFICATION WORK ===
@dataclass
class WorkItem:
    kind: str          # 'ids', 'x', or 'y'
    src_path: str      # Input file path (mask or dense)
    out_path: str      # Output sparse file path
    ids_path: str      # Path to ids-sparse file ('' for ids tasks)
    label: str         # For logging purposes

# === HELPERS ===
def get_all_files_recursive(root, filter_fn=None):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if filter_fn is None or filter_fn(f):
                yield os.path.join(dirpath, f)

def log_work_summary(items: List[WorkItem], kind: str):
    print(f"\n{kind.upper()} FILES:")
    to_do = [item for item in items if overwrite or not os.path.exists(item.out_path)]
    to_skip = [item for item in items if not overwrite and os.path.exists(item.out_path)]
    print(f"  Will sparsify: {len(to_do)}")
    print(f"  Skipped (already exists): {len(to_skip)}")
    return to_do

def process_work_items(items: List[WorkItem]):
    for i, item in enumerate(items, 1):
        if item.kind == 'ids':
            mask = pd.read_csv(item.src_path, header=None).values
            row_sums = np.sum(mask, axis=1)
            pos_out = np.zeros((mask.shape[0], int(row_sums.max())), dtype=int)
            for r in range(mask.shape[0]):
                idxs = np.where(mask[r] == 1)[0]
                pos_out[r, :len(idxs)] = idxs + 1  # 1-based
            pd.DataFrame(pos_out).to_csv(item.out_path, header=False, index=False)
        else:
            dense = pd.read_csv(item.src_path, header=None).values
            pos = pd.read_csv(item.ids_path, header=None).values
            sparse = np.zeros_like(pos, dtype=float)
            for r in range(pos.shape[0]):
                idxs = pos[r][pos[r] > 0] - 1  # back to 0-based
                sparse[r, :len(idxs)] = dense[r, idxs]
            pd.DataFrame(sparse).to_csv(item.out_path, header=False, index=False)

        print(f"[{i}/{len(items)}] {item.kind.upper()} sparsified: {item.label}")

# === STAGE 1: GATHER TASKS ===

skipped_due_to_missing_ids = []

# Find mask files (for ids-sparse)
mask_files = list(get_all_files_recursive(rootdir, lambda f: f.startswith(mask_prefix)))
ids_tasks = []
for mask_path in mask_files:
    suffix = os.path.basename(mask_path).replace(mask_prefix, '')
    ids_path = os.path.join(rootdir, f"{ids_prefix}{suffix}")
    ids_tasks.append(WorkItem('ids', mask_path, ids_path, '', f"ids-sparse_{suffix}"))

# Find x0 dense files
x_dense_files = [f for f in os.listdir(rootdir) if f.startswith(x_dense_prefix)]
x_tasks = []
for f in x_dense_files:
    suffix = f.replace(x_dense_prefix, '')
    dense_path = os.path.join(rootdir, f)
    sparse_path = os.path.join(rootdir, f"{x_sparse_prefix}{suffix}")
    ids_path = os.path.join(rootdir, f"{ids_prefix}{suffix}")
    if os.path.exists(ids_path):
        x_tasks.append(WorkItem('x', dense_path, sparse_path, ids_path, f"x0_{suffix}"))
    else:
        skipped_due_to_missing_ids.append((f"x0_{suffix}", ids_path))

# Find y dense files (recursive, with variable prefix)
y_dense_files = list(get_all_files_recursive(rootdir, lambda f: y_dense_suffix_pattern.match(f)))
y_tasks = []
for y_path in y_dense_files:
    match = y_dense_suffix_pattern.match(os.path.basename(y_path))
    if match:
        prefix, suffix = match.group(1), match.group(2)
        ids_path = os.path.join(rootdir, f"{ids_prefix}{suffix}.csv")
        sparse_name = y_sparse_template.format(prefix, suffix)
        y_dir = os.path.dirname(y_path)
        sparse_path = os.path.join(y_dir, sparse_name)
        if os.path.exists(ids_path):
            y_tasks.append(WorkItem('y', y_path, sparse_path, ids_path, f"{prefix}_y_{suffix}"))
        else:
            skipped_due_to_missing_ids.append((f"{prefix}_y_{suffix}", ids_path))


# === STAGE 2: LOG SUMMARY BEFORE EXECUTION ===
print("\n=== PRE-SPARSIFICATION REPORT ===")
ids_todo = log_work_summary(ids_tasks, 'ids')
x_todo = log_work_summary(x_tasks, 'x')
y_todo = log_work_summary(y_tasks, 'y')

if skipped_due_to_missing_ids:
    print("\n⚠️ FILES SKIPPED due to missing ids-sparse files:")
    for label, expected_path in skipped_due_to_missing_ids:
        print(f" - {label} (missing: {expected_path})")

# === STAGE 3: EXECUTE SPARSIFICATION ===
print("\n=== STARTING SPARSIFICATION ===")
process_work_items(ids_todo)
process_work_items(x_todo)
process_work_items(y_todo)

print("\nSparsification complete.")
