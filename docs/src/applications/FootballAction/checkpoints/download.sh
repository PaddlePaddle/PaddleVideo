# audio
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/FootballAction/audio.tar
# pptsm
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/FootballAction/pptsm.tar
# bmn
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/FootballAction/bmn.tar
# lstm
wget https://videotag.bj.bcebos.com/PaddleVideo-release2.1/FootballAction/lstm.tar

tar -xvf audio.tar
tar -xvf pptsm.tar
tar -xvf bmn.tar
tar -xvf lstm.tar

rm -f audio.tar
rm -f pptsm.tar
rm -f bmn.tar
rm -f lstm.tar
