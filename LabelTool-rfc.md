功能名称： 为 PaddleVideo 新增视频数据标注工具

开始日期：（填写提交时候的日期，2021-10-20）

RFC PR：

GitHub Issue：[PaddleVideo#203](https://github.com/PaddlePaddle/PaddleVideo/issues/203)   

# 总结
为 PaddleVideo 新增视频数据标注工具。按照Kinetics数据集的格式，记录视频的label, youtube_id(file_name), time_start, time_end, split等信息，并保存为csv格式。   
整理UI类似于labelme，右边整体为file_list, 中间分为三部分:   
视频播放VideoWidget, 一个进度条，以及一组按钮用来记录起止时间和类别。   

# 动机
参考了一些剪辑软件，设计出这种简单地容易操作的UI.   

# 使用指南级别的说明
使用类似于labelme, 包含最基础的打开、保存等功能，其次实现标注工具的功能。
在底部操作按钮部分：      
通过GetXXXTime，可以快速的从进度条中获取时间，并可以通过微调按钮进行修正；   通过SetXXXTime, 可以将视频跳到我们指定的时间上，方便我们对标注的时间进行验证。   
使用combobox进行快速的选择标签label，并可以勾选数据属于train or test。   

# 参考文献级别的说明
可以方便的获取时间，根据的是mediaplayer的数据；同样快捷的设置时间用于验证，也是基于同样的原因。   

# 缺点
无   

# 理论依据和替代方案
参考了格式工厂软件中对视频剪裁的部分。将整体的操作UI放置在画面下方。   
放在侧方不美观。   

# 现有技术
PYQT已经非常成熟，制作这样的标注工具并不是非常困难。   
labelme稍微复杂了一点，可以从labelImg上手更快。   

# 未解决的问题

没有决定是否还保留labelme or labelImg中的canvas部分。   

# 未来的可能性
未来针对视频的别的标注任务仍然可以在此基础上进行扩展。   
