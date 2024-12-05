# Model Evaluation Tool
A simple tool for manually evaluating the performance of your Groundlight Binary ML model.

This script provides a simple way for you to do an independent evaluation of your Groundlight model's ML performance. Note that this is not the recommended way of using our service, as this only evaluates ML performance and not the combined performance of our ML + escalation system. However, the balanced accuracy results from `evaluate.py` should fall within the bounds of Projected ML Accuracy shown on our website, if the train and evaluation dataset that the user provided are well randomized.

Note this tool only works for **binary detectors**.

## Installation

The dependencies for this script can be installed by either using poetry (recommended) or `requirements.txt`.

Using poetry

```bash
poetry install
```

Using `requirements.txt`
```bash
pip install -r requirements.txt
```

## Usage

### Setting Up Your Account

To train a ML model, make sure to create a binary detector on the [Online Dashboard](https://dashboard.groundlight.ai/).

You will also need to create an API Token to start uploading images to the account. You can go [here](https://dashboard.groundlight.ai/reef/my-account/api-tokens) to create one.

After you have created your API token, add the token to your terminal as an variable:

```bash
export GROUNDLIIGHT_API_TOKEN="YOUR_API_TOKEN"
```

### Formatting Dataset

This script assumes your custom image dataset is structured in the following format:

```bash
└── dataset
    ├── dataset.csv
    └── images
        ├── 1.jpg
        ├── 10.jpg
        ├── 11.jpg
        ├── 12.jpg
        ├── 13.jpg
        ├── 14.jpg
```

The `dataset.csv` file should have two columns: image_name and label (YES/NO), for example:

```bash
1.jpg,YES
11.jpg,NO
12.jpg,YES
13.jpg,YES
14.jpg,NO
```

The corresponding image file should be placed inside the `images` folder.

### Training the Detector

To train the ML model for a detector, simply run the script `train.py` with the following arguments:

```bash
poetry run python train.py --detector-name NAME_OF_THE_DETECTOR --detector-query QUERY_OF_THE_DETECTOR --dataset PATH_TO_DATASET_TRAIN_FOLDER
```

Optionally, set the `--delay` argument to prevent going over the throttling limit of your account.

### Evaluate the Detector

Before evaluating the ML model, you should wait a few minutes for the model to be fully trained.  Small models generally train very quickly, but to be sure your model is fully trained, you should wait 10 or 15 minutes after submitting the training images.

To evaluate the ML model performance for a detector, simply run the script `evaluate.py` with the following arguments:

```bash
poetry run python evaluate.py --detector-id YOUR_DETECTOR_ID --dataset PATH_TO_DATASET_TEST_FOLDER
```

Optionally, set the `--delay` argument to prevent going over the throttling limit of your account.

The evaluation script will output the following information:

```
Number of Correct ML Predictions
Average Confidence
Balanced Accuracy
Precision
Recall
```

## Questions?

If you have any questions or feedback about this tool, feel free to reach out to your Groundlight contact, over email at support@groundlight.ai, in your dedicated Slack channel, or using the chat widget in the bottom-right corner of the dashboard.