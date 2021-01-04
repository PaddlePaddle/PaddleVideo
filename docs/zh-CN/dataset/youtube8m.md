YouTube-8M 是一个大规模视频标签数据集，包含百万级视频，具有高质量机器生成的标注，涵盖3800+实体。  
这里用到的是YouTube-8M 2018年更新之后的数据集。  
我们将下载的TFRecord文件转化为pickle文件以便PaddlePaddle使用。  

数据下载  
请使用Youtube-8M官方链接分别下载训练集（http://us.data.yt8m.org/2/frame/train/index.html ） 和  
验证集（http://us.data.yt8m.org/2/frame/validate/index.html ）。  
每个链接里各提供了3844个文件的下载地址，用户也可以使用官方提供的下载脚本下载数据。  
数据下载完成后，将会得到3844个训练数据文件和3844个验证数据文件（TFRecord格式）。   

数据格式转化  
为了加速，需要将TFRecord文件格式转成了pickle格式，请使用转化脚本：tf2pkl.py。  
然后将pkl拆分为单视频一个文件，请使用拆分脚本：split_yt8m.py。  
（ https://github.com/PaddlePaddle/PaddleVideo/blob/main/data/yt8m/split_yt8m.py ）
