import os
from tcia_utils import nbia

print("Fetching series list for C4KC-KiTS...")
series_json = nbia.getSeries(collection='C4KC-KiTS')

if series_json:
    print(f"Found {len(series_json)} series. Starting download...")
    os.makedirs('datasets/dataset_3_masks', exist_ok=True)
    # Download the series
    nbia.downloadSeries(series_json, path='datasets/dataset_3_masks')
    print("Download completed successfully!")
else:
    print("No series found for C4KC-KiTS!")
