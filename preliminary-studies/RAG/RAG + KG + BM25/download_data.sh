#!/bin/bash
set -e
set -x

pip install gdown

ZIP_NAME="musique_v1.0.zip"
gdown 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h -O $ZIP_NAME
unzip $ZIP_NAME
rm $ZIP_NAME
rm -rf __MACOSX