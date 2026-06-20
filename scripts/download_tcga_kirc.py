import tcia_utils.nbia as nbia
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Fetching series for TCGA-KIRC...")
    try:
        series = nbia.getSeries(collection='TCGA-KIRC')
        print(f"Found {len(series)} series.")
    except Exception as e:
        print(f"Error fetching series: {e}")
        return

    print("Starting download. This will take a long time and download ~91GB of data.")
    try:
        # Download all series in the collection
        nbia.downloadSeries(series, path='/home/administrator/Desktop/RCC/datasets/dataset_3_TCGA-KIRC')
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading series: {e}")

if __name__ == "__main__":
    main()
