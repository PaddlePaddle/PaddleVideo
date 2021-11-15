执行
```bash
bash ./run_all.sh down_data
```
即可运行.

run_all.sh内部的执行步骤：
1. cd 到 ../../ (也就是 PaddleVideo 目录)
2. 切换到benchmark_dev分支
3. 安装 PaddleVideo 所需依赖
4. cd 回PaddleVideo/data/ucf101
5. wget下载数据集并解压缩，并下载预训练权重放到data目录下
6. 再次cd 回到 ../../ (也就是 PaddleVideo 目录)
8. 按照不同的参数执行 run_benchmark.sh 脚本
