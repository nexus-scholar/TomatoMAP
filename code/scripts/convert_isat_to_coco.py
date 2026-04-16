import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.isat_to_coco import main


if __name__ == "__main__":
    main()
