from abc import abstractmethod
from ... import builder
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BaseRecognizer(nn.Layer):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``train_step``, supporting to forward when training.
    - Methods:``valid_step``, supporting to forward when validating.
    - Methods:``test_step``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        head (dict): Classification head to process feature.

    """
    def __init__(self, backbone=None, head=None):

        super().__init__()
        if backbone!=None:
            self.backbone = builder.build_backbone(backbone)
            self.backbone.init_weights()
        else:
            self.backbone = None
        if head!=None:
            self.head_name = head.name
            self.head = builder.build_head(head)
            self.head.init_weights()
        else:
           self.head = None


    def init_weights(self):
        """Initialize the model network weights. """

        self.backbone.init_weights(
        )  #TODO: required? while backbone without base class
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

    def forward(self, imgs, **kwargs):
        """Define how the model is going to run, from input to output.
        """
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = paddle.reshape(imgs, [-1] + list(imgs.shape[2:]))

        if self.backbone!=None:
            feature = self.extract_feature(imgs)
        else:
            feature = imgs
        if self.head!=None:
            cls_score = self.head(feature, num_segs)
        else:
            cls_score = None


        return cls_score

    @abstractmethod
    def train_step(self, data_batch, **kwargs):
        """Training step.
        """
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data_batch, **kwargs):
        """Validating step.
        """
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data_batch, **kwargs):
        """Test step.
        """
        raise NotImplementedError
