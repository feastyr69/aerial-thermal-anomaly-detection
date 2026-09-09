#!/bin/bash

# Check if credentials are provided
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set."
    echo "You can get these from your kaggle.json file (Account settings -> Create New API Token)."
    echo ""
    echo "Usage:"
    echo "  KAGGLE_USERNAME=your_username KAGGLE_KEY=your_key ./download_flir.sh"
    exit 1
fi

echo "Downloading FLIR dataset from Kaggle using cURL..."
mkdir -p ./raw_flir

# Use curl with basic auth to hit the Kaggle API
curl -L -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
     -o ./raw_flir/dataset.zip \
     "https://www.kaggle.com/api/v1/datasets/download/deepnewbie/flir-thermal-images-dataset"

echo "Unzipping dataset..."
cd ./raw_flir
unzip -q dataset.zip
rm dataset.zip
cd ..

echo "Download and extraction complete!"

