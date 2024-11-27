#!/usr/bin/env python3
"""
A script to upload frames with labels to a detector in a controlled manner.
You can specify the delay between uploads.
"""

import argparse
import os
import PIL
import time
import PIL.Image
import pandas as pd
import logging

from groundlight import Groundlight, Detector
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def upload_image(gl: Groundlight, detector: Detector, image: PIL, label: str) -> None:
    """
    Upload a image with a label to a detector.

    Args:
        gl: The Groundlight object.
        detector: The detector to upload to.
        image: The image to upload.
        label: The label to upload.
    """
    
    # Convert image to jpg if not already
    if image.format != "JPEG":
        image = image.convert("RGB")

    if label not in ["YES", "NO"]:
        raise ValueError(f"Invalid label: {label}, must be 'YES' or 'NO'.")

    # Use ask_ml to upload the image and then add the label to the image query
    iq = gl.ask_ml(detector=detector, image=image)
    gl.add_label(image_query=iq, label=label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload images with labels to a detector.")
    parser.add_argument("--detector-id", type=str, required=True, help="The ID of the detector to upload to.")
    parser.add_argument("--dataset", type=str, required=True, help="The folder containing the dataset.csv and images folder")
    parser.add_argument("--delay", type=float, required=False, default=0.1, help="The delay between uploads.")
    args = parser.parse_args()

    gl = Groundlight()
    detector = gl.get_detector(args.detector_id)

    # Load the dataset from the CSV file and images from the images folder
    # The CSV file should have two columns: image_name and label (YES/NO)
    
    dataset = pd.read_csv(os.path.join(args.dataset, "dataset.csv"))
    images = os.listdir(os.path.join(args.dataset, "images"))
    
    logger.info(f"Uploading {len(dataset)} images to detector {detector.name} with delay {args.delay}.")
    
    for image_name, label in tqdm(dataset.values):
        if image_name not in images:
            logger.warning(f"Image {image_name} not found in images folder.")
            continue

        image = PIL.Image.open(os.path.join(args.dataset, "images", image_name))
        upload_image(gl=gl, detector=detector, image=image, label=label)
        time.sleep(args.delay)
    
    logger.info("Upload complete. Please wait around 10 minutes for the detector to retrain.")
