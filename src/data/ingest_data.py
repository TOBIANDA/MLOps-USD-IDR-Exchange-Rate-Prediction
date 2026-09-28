"""
Re-export and CLI proxy for src/ingest_data.py under src/data/.
Ensures compatibility whether scripts are called from src/ or src/data/.
"""
import sys
from pathlib import Path

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest_data import IngestionWorker, ingest_usd_idr_rates, parse_args

if __name__ == "__main__":
    args = parse_args()
    formats = ["csv", "json"] if args.format == "all" else [args.format]

    worker = IngestionWorker(raw_dir=args.output_dir)
    df, paths = worker.run(
        days=args.days,
        currency=args.currency,
        export_formats=formats,
        sync_db=not args.no_db_sync,
        use_timestamp=not args.no_timestamp,
    )

    print("\n" + "=" * 60)
    print(" [INGESTION SUCCESS] Bank Indonesia Data Ingestion Summary")
    print("=" * 60)
    print(f"Total Records Ingested : {len(df)}")
    if not df.empty:
        print(f"Date Coverage Range    : {df['tgl'].min()} to {df['tgl'].max()}")
        print(f"Latest Rate (Beli/Jual): IDR {df.iloc[-1]['beli']:,.2f} / IDR {df.iloc[-1]['jual']:,.2f}")
    print("\nSaved Output Files:")
    for key, path in paths.items():
        print(f"  • {key.upper()}: {path}")
    print("=" * 60 + "\n")
