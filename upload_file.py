import argparse
from utils.tools import upload_to_gcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    upload_to_gcs(args.filename)
