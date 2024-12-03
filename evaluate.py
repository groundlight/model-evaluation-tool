#!/usr/bin/env python3
"""
A script to evaluate the accuracy of a detector on a given dataset.
It will upload the images to the detector and compare the predicted labels with the ground truth labels.
You can specify the delay between uploads.
"""

import argparse
import os
import PIL
import time
import PIL.Image
import pandas as pd
import logging

from groundlight import Groundlight, Detector, BinaryClassificationResult
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def upload_image(gl: Groundlight, detector: Detector, image: PIL) -> BinaryClassificationResult:
    """
    Upload a image with a label to a detector.

    Args:
        gl: The Groundlight object.
        detector: The detector to upload to.
        image: The image to upload.
    Returns:
        The predicted label (YES/NO).
    """

    # Convert image to jpg if not already
    if image.format != "JPEG":
        image = image.convert("RGB")

    # Use ask_ml to upload the image and then return the result
    iq = gl.ask_ml(detector=detector, image=image)
    return iq.result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the accuracy of a detector on a given dataset.")
    parser.add_argument("--detector-id", type=str, required=True, help="The ID of the detector to evaluate.")
    parser.add_argument("--dataset", type=str, required=True, help="The folder containing the dataset.csv and images folder")
    parser.add_argument("--delay", type=float, required=False, default=0.1, help="The delay between uploads.")
    args = parser.parse_args()

    gl = Groundlight()
    detector = gl.get_detector(args.detector_id)

    # Load the dataset from the CSV file and images from the images folder
    # The CSV file should have two columns: image_name and label (YES/NO)

    dataset = pd.read_csv(os.path.join(args.dataset, "dataset.csv"))
    images = os.listdir(os.path.join(args.dataset, "images"))

    logger.info(f"Evaluating {len(dataset)} images on detector {detector.name} with delay {args.delay}.")

    # Record the number of correct predictions
    # Also record the number of TP, TN, FP, FN for calculating balanced accuracy, precision, and recall
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_processed = 0
    average_confidence = 0

    for image_name, label in tqdm(dataset.values):
        if image_name not in images:
            logger.warning(f"Image {image_name} not found in images folder.")
            continue

        if label not in ["YES", "NO"]:
            logger.warning(f"Invalid label {label} for image {image_name}. Skipping.")
            continue

        image = PIL.Image.open(os.path.join(args.dataset, "images", image_name))
        result = upload_image(gl=gl, detector=detector, image=image)

        if result.label == "YES" and label == "YES":
            true_positives += 1
        elif result.label == "NO" and label == "NO":
            true_negatives += 1
        elif result.label == "YES" and label == "NO":
            false_positives += 1
        elif result.label == "NO" and label == "YES":
            false_negatives += 1

        average_confidence += result.confidence
        total_processed += 1

        time.sleep(args.delay)

    # Calculate the accuracy, precision, and recall
    balanced_accuracy = (true_positives / (true_positives + false_negatives) + true_negatives / (true_negatives + false_positives)) / 2
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    logger.info(f"Processed {total_processed} images.")
    logger.info(f"Average Confidence: {average_confidence / total_processed:.2f}")
    logger.info(f"Balanced Accuracy: {balanced_accuracy:.2f}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
