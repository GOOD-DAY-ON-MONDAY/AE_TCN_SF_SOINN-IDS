import os
import argparse
import subprocess
import shutil

REPO_URL = "https://github.com/ACANETS/NetML-Competition2020.git"
BASE_DIR = os.path.join("data", "raw")

DATASET_PATHS = {
    "netml2020": "data/NetML",
    "cicids2017": "data/CICIDS2017"
}

def download_dataset(dataset_name):
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_PATHS.keys())}")

    target_dir = os.path.abspath(os.path.join(BASE_DIR, dataset_name))
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"=== Downloading {dataset_name} ===")
    
    tmp_dir = os.path.abspath(os.path.join(BASE_DIR, f"_tmp_{dataset_name}"))
    
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, tmp_dir], check=True)
        git_target = DATASET_PATHS[dataset_name]
        subprocess.run(["git", "sparse-checkout", "set", git_target], cwd=tmp_dir, check=True)
        src_path = os.path.join(tmp_dir, git_target)
        for item in os.listdir(src_path):
            s = os.path.join(src_path, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

        print(f"Successfully saved to {target_dir}")

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for NetML project.")
    parser.add_argument("--dataset", type=str, required=True, choices=["netml2020", "cicids2017", "all"])
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASET_PATHS.keys():
            download_dataset(ds)
    else:
        download_dataset(args.dataset)