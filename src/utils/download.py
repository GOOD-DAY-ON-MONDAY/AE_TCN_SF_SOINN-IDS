import argparse
import subprocess
import shutil
from pathlib import Path

REPO_URL = "https://github.com/ACANETS/NetML-Competition2020.git"
BASE_DIR = Path("data/raw")

DATASET_PATHS = {
    "netml2020": "data/NetML",
    "cicids2017": "data/CICIDS2017"
}

def download_dataset(dataset_name: str, force: bool = False):
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_PATHS.keys())}")

    # Use pathlib for clean path resolution
    target_dir = (BASE_DIR / dataset_name).resolve()
    
    print(f"=== Downloading {dataset_name} from {REPO_URL} ===")
    
    # Pathlib methods now work correctly on the Path object
    if target_dir.exists() and not force:
        if target_dir.is_dir() and any(target_dir.iterdir()):
            print(f"[SKIP] '{dataset_name}' already exists at {target_dir} — skipping download.")
            return
        elif target_dir.is_file():
            print(f"[SKIP] '{dataset_name}' already exists at {target_dir} — skipping download.")
            return
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = (BASE_DIR / f"_tmp_{dataset_name}").resolve()
    
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(tmp_dir)], check=True)
        git_target = DATASET_PATHS[dataset_name]
        subprocess.run(["git", "sparse-checkout", "set", git_target], cwd=str(tmp_dir), check=True)
        
        src_path = tmp_dir / git_target
        
        for item in src_path.iterdir():
            d = target_dir / item.name
            if item.is_dir():
                if d.exists():
                    shutil.rmtree(d)
                shutil.copytree(item, d)
            else:
                shutil.copy2(item, d)

        print(f"Successfully saved to {target_dir}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git operation failed: {e}")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for NetML project.")
    parser.add_argument("--dataset", type=str, required=True, choices=["netml2020", "cicids2017", "all"])
    parser.add_argument("--force", action="store_true", help="Re-download even if data already exists.")
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in DATASET_PATHS.keys():
            download_dataset(ds, force=args.force)
    else:
        download_dataset(args.dataset, force=args.force)