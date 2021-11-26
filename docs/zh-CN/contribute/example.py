import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
import paddle.nn as nn


# 1. 数据加载和处理
## 1.1 数据处理Pipeline
class ExamplePipeline(object):
    """ Example Pipeline"""
    def __init__(self, mean=0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        data = results['data']
        norm_data = (data - self.mean) / self.std
        results['data'] = norm_data
        return results


class ExampleDataset(Dataset):
    """ExampleDataset"""
    def __init__(self):
        super(ExampleDataset, self).__init__()
        self.x = np.random.rand(100, 20, 20)
        self.y = np.random.randint(10, size=(100, 1))

    def __getitem__(self, idx):
        x_item = self.x[idx]
        results = {}
        results['data'] = x_item
        pipeline = ExamplePipeline()
        results = pipeline(results)
        x_item = results['data'].astype('float32')
        y_item = self.y[idx].astype('int64')
        return x_item, y_item

    def __len__(self):
        return self.x.shape[0]


train_dataset = ExampleDataset()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)


# 2. 网络
class ExampleModel(nn.Layer):
    """Example Model"""
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.layer1 = paddle.nn.Flatten(1, -1)
        self.layer2 = paddle.nn.Linear(400, 512)
        self.layer3 = paddle.nn.ReLU()
        self.layer4 = paddle.nn.Dropout(0.2)
        self.layer5 = paddle.nn.Linear(512, 10)

    def forward(self, x):
        """ model forward"""
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.layer5(y)
        return y


model = ExampleModel()
model.train()

# 3. 优化器
optim = paddle.optimizer.Adam(parameters=model.parameters())

epochs = 5
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)

        loss = paddle.nn.functional.cross_entropy(predicts, y_data)

        # 4. 评估指标
        acc = paddle.metric.accuracy(predicts, y_data)

        loss.backward()
        print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
            epoch, batch_id, loss.numpy(), acc.numpy()))

        optim.step()
        optim.clear_grad()
