cd data/ntu-rgb-d

# download
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H" -O nturgbd_skeletons_s001_to_s017.zip && rm -rf /tmp/cookies.txt

unzip nturgbd_skeletons_s001_to_s017.zip && rm -rf nturgbd_skeletons_s001_to_s017.zip

wget https://videotag.bj.bcebos.com/Data/statistics.zip

mkdir statistics

unzip statistics.zip -d statistics/ && rm -rf statistics.zip
