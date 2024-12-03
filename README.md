# Model Evaluation Tool
A simple tool for evaluating the performance of your Groundlight Binary ML model.

This script provides a simple way for users to do an independent evaluation of the ML's performance. Note that this is not the recommended way of using our service, as this only evaluates ML performance and not the combined performance of our ML + escalation system. However, the balanced accuracy results from `evaluate.py` should fall within the bounds of Projected ML Accuracy shown on our website, if the train and evaluation dataset that the user provided are well randomized.

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

To evaluate the ML model performance for a detector, simply run the script `evaluate-accuracy.py` with the following arguments:

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