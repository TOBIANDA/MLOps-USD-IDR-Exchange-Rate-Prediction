"""
Re-export and CLI proxy for src/preprocess.py under src/data/.
Ensures compatibility whether scripts are called from src/ or src/data/.
"""
import sys
from pathlib import Path

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import Preprocessor, preprocess_usd_idr_data, parse_args

if __name__ == "__main__":
    args = parse_args()
    preprocessor = Preprocessor(processed_dir=args.output_dir)
    clean_df, paths = preprocessor.run(
        input_file=args.input,
        output_filename=args.output_file,
    )

    print("\n" + "=" * 60)
    print(" [PREPROCESSING SUCCESS] USD/IDR Data Cleaning Summary")
    print("=" * 60)
    print(f"Clean Records Generated : {len(clean_df)}")
    if not clean_df.empty:
        print(f"Date Range Cleaned      : {clean_df['tgl'].min()} to {clean_df['tgl'].max()}")
        print(f"Current Kurs Tengah Mean: IDR {clean_df['tengah'].mean():,.2f}")
        print(f"Average Spread          : IDR {clean_df['spread'].mean():,.2f}")
    print("\nProcessed Output Files:")
    for key, path in paths.items():
        print(f"  • {key.upper()}: {path}")
    print("=" * 60 + "\n")
