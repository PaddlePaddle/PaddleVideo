#! /usr/bin/bash env

wget --no-check-certificate "https://videotag.bj.bcebos.com/Data/ucf24.zip"
unzip -q ucf24.zip
rm -rf ./ucf24.zip
