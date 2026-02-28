import shutil
from pathlib import Path
import kagglehub

DATASET = "redwankarimsony/heart-disease-data"

def main():
    # Download dataset to kagglehub cache, returns a local folder path
    cache_dir = kagglehub.dataset_download(DATASET)
    print(f"Dataset downloaded to cache: {cache_dir}")

    # Copy into the data/raw folder
    dest_dir = Path("data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in Path(cache_dir).glob("*.csv"): 
        shutil.copy(file, dest_dir / file.name)
        print("Copied:", file.name)
        
    print("Dataset ready in data/raw folder.")
    
if __name__ == "__main__":
    main()