YouTube-8M is a large-scale labeled video dataset that consists of millions of YouTube video IDs,   
with high-quality machine-generated annotations from a diverse vocabulary of 3,800+ visual entities.  
we use YouTube-8M 2018 , which is Youtube-8M’s update version in May 14th, 2018,   
with improved quality machine-generated labels, and reduced size / higher-quality video dataset.  
We transform TFRecord to pickle format for PaddleVideo.  

download  
Please use Youtube-8M’s official link to download training setand validate set。  
There’s 3844 files’ download link for each set. official download tools also can be used.   
When the download is finished, you’ll get 3844 training files and 3844 validate files（TFRecord format）。  

Transform Data Format  
To speed up，we transform TFRecord to pickle format using tf2pkl.py,   
and split pkl to single video file using split_yt8m.py.  
（ https://github.com/PaddlePaddle/PaddleVideo/blob/main/data/yt8m/split_yt8m.py ）
