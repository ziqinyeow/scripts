#!/bin/bash

curl -L -o ./download.py https://raw.githubusercontent.com/ziqinyeow/scripts/main/rsna/download.py
python -c "from download import download_dataset; download_dataset()"
rm ./download.py