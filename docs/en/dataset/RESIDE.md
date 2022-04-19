# RESIDE-Standard data preparation

Preparation of RESIDE-Standard data. It mainly includes the download of three sub-datasets contained in the RESIDE-Standard dataset.



## Data Download

Details of the data can be found in **dataset address:** [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/) (Please select the [RESIDE-Standard dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) in the URL). The specific download links are given below:

**Dataset download address：**

**ITS (Indoor Training Set)：**http://tinyurl.com/yaohd3yv                                    **Passward**:  g0s6

**OTS：**  https://pan.baidu.com/s/1c2rW4hi                                                             **Passward**:  5vss

**SOTS ：** https://pan.baidu.com/share/init?surl=SSVzR058DX5ar5WL5oBTLg  **Passward**:  s6tu



## After the above steps are completed, the file organization format is as follows

**file structure**


```
|-- data/RESIDE
	|-- ITS
		|-- hazy
			|-- *.png
		|-- clear
			|-- *.png
	|-- OTS
		|-- hazy
			|-- *.jpg
		|-- clear
			|-- *.jpg
	|-- SOTS
		|-- indoor
			|-- hazy
				|-- *.png
			|-- clear
				|-- *.png
		|-- outdoor
			|-- hazy
				|-- *.jpg
			|-- clear
				|-- *.png
```

