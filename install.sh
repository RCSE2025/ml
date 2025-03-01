#!/bin/sh

apt update
apt upgrade -y
apt install ffmpeg libsm6 libxext6 -y
apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng -y
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt