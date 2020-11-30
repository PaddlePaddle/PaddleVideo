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


    @abstractmethod
    def forward_train(self, imgs, labels, **kwargs):
        pass

    @abstractmethod
    def forward_valid(self, imgs):
        pass


    def forward(self, imgs, labels=None, return_loss=True, reducesum=False, **kwargs):
        """Define how the model is going to run, from input to output.
        """
        if return_loss:
            if labels is None:
                raise ValueError("Label should not be None.")
            return self.forward_train(imgs, labels, reducesum=reducesum, **kwargs)
        else:
            return self.forward_valid(imgs)

    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        imgs = data_batch[0]
        labels = data_batch[1]

        # call forward
        loss_metrics = self(imgs, labels, return_loss=True)
        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        """Validating setp.
        """
        imgs = data_batch[0]
        labels = data_batch[1]

        # call forward
        loss_metrics = self(imgs, labels, reducesum=True, return_loss=True)
        return loss_metrics
