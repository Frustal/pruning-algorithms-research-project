import shutil
from pathlib import Path

def clear_outputs(root_dir="output"):
    """Deletes logs and models folders to start fresh."""
    root = Path(root_dir)
    for folder in ["logs", "models"]:
        path = root / folder
        if path.exists():
            print(f"Deleting {path}...")
            shutil.rmtree(path)

if __name__ == '__main__':
    clear_outputs()