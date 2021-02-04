RNN is usualy used for sequence data, it works well for videos’ time-step information of continues frames.  
 And it’s a basic popular method for video classification.  
 AttentionLstm use a bidirectional LSTM to encode the frames of video.  
 Different from the traditional method where LSTM is in the output layer for the last time,  
 AttentionLstm increase a Attention layer, there is a adaptive weight for the hidden layer of each time,  
 and weighted to get the final feature vector.  
 For Attention layer, refer to the paper: AttentionCluster.  

Data Prepare, refer to the page of dataset: (https://github.com/PaddlePaddle/PaddleVideo/blob/main/docs/en/dataset/youtube8m.md)  
Model training and valid, refer to the page of tutorial: (TODO)  
The Hit@1 on Youtube-8M’ validation is 0.89, PERR is	0.8012, GAP is 0.8594.
