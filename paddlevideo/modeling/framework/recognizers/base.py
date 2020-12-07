from abc import abstractmethod
from ... import builder
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BaseRecognizer(nn.Layer):
    """Base class for recognizers.
       
    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_valid``, supporting to forward when validating.

    Args:
        backbone (dict): Backbone modules to extract feature.
        head (dict): Classification head to process feature.
        #XXX cfg keep or not????
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None. 
 
    """
    def __init__(self,
		 backbone,
		 head):

        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)
        self.init_weights()


    def init_weights(self):
        """Initialize the model network weights. """
        
        self.backbone.init_weights()
        self.head.init_weights()

    def extract_feature(self, imgs):
        """Extract features through a backbone.
	
	Args:
	    imgs (paddle.Tensor) : The input images.

        Returns:
	    feature (paddle.Tensor) : The extracted features.
        """
        feature = self.backbone(imgs)
        return feature


    def average_clip(self, cls_score, num_segs=1):
        batch_size = cls_score.shape[0]
        cls_score = paddle.reshape(batch_size // num_segs, num_segs, -1)
        softmax_out = F.softmax(cls_score)
        return softmax_out


    @abstractmethod
    def forward_train(self, imgs, labels, reduce_sum, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, imgs, labels,reduce_sum, **kwargs):
        pass


    def forward(self, imgs, **kwargs):
        """Define how the model is going to run, from input to output.
        """
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = paddle.reshape(imgs, [-1]+list(imgs.shape[2:]))

        feature = self.extract_feature(imgs)
        cls_score = self.head(feature, num_segs)

        return cls_score


    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        imgs = data_batch[0]
        labels = data_batch[1]

        # call forward
        loss_metrics = self.forward_train(imgs, labels, reduce_sum=False, **kwargs)

        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        """Validating setp.
        """
        imgs = data_batch[0]
        labels = data_batch[1]

        # call forward
        loss_metrics = self.forward_train(imgs, labels, reduce_sum=True, **kwargs)
        return loss_metrics

    def test_step(self, data_batch, **kwargs):
        imgs = data_batch[0]
        labels = data_batch[1]

        metrics = self.forward_test(imgs, labels, reduce_sum=True, **kwargs)
        return metrics
        
