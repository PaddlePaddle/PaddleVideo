"""This module defines the TextEmbedding interface for converting video descriptions and
queries into embeddings.
"""
import zipfile
import functools
from abc import abstractmethod
from pathlib import Path

import numpy as np
import paddle
import gensim
import requests
import transformers
from typeguard import typechecked
from zsvision.zs_utils import BlockTimer

from model.s3dg import S3D

class TextEmbedding:
    def __init__(self, model, dim: int):
        self.model = model
        self.dim = dim
        #self.device = None

    @abstractmethod
    def text2vec(self, text: str) -> np.ndarray:
        """Convert a string of text into an embedding.

        Args:
            text: the content to be embedded

        Returns:
            (d x n) array, where d is the dimensionality of the embedding and `n` is the
                number of words that were successfully parsed from the text string.

        NOTE: For some text embedding models (such as word2vec), not all words are
        converted to vectors (e.g. certain kinds of stop words) - these are dropped from
        the output.
        """
        raise NotImplementedError

    #@typechecked
    #def set_device(self, device: torch.device):
    #    self.model = self.model.to(device)
    #    self.device = device


@functools.lru_cache(maxsize=64, typed=False)
def load_w2v_model_from_cache(
        w2v_weights: Path,
) -> gensim.models.keyedvectors.Word2VecKeyedVectors:
    with BlockTimer("Loading w2v from disk"):
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fname=w2v_weights,
            binary=True,
        )
    return model


@typechecked
def fetch_model(url: str, weights_path: Path):
    weights_path.parent.mkdir(exist_ok=True, parents=True)
    with BlockTimer(f"Fetching weights {url} -> {weights_path}"):
        resp = requests.get(url, verify=False)
        with open(weights_path, "wb") as f:
            f.write(resp.content)


class W2VEmbedding(TextEmbedding):
    """This model embeds text using the google-released implementation of the word2vec
    model introduced in:

        Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
        Distributed representations of words and phrases and their compositionality.
        In Advances in neural information processing systems (pp. 3111-3119).

    For words that are present in the w2v vocabulary, a 300-dimensional embedding is
    produced via a lookup table.
    """
    @typechecked
    def __init__(
            self,
            dim: int,
            mirror: str,
            weights_path: Path,
            fetch_weights: bool = True,
    ):
        if not weights_path.exists():
            if fetch_weights:
                fetch_model(url=mirror, weights_path=weights_path)
            else:
                raise ValueError(f"w2v weights missing at {weights_path}")

        model = load_w2v_model_from_cache(weights_path)
        super().__init__(model=model, dim=dim)

    @typechecked
    def text2vec(self, text: str) -> np.ndarray:
        # convert the text string to tokens that can be processed by w2v.  We handle
        # 'a' as a special case.
        tokens = [x for x in text.split(" ") if x != "a" and x in self.model.vocab]

        embeddings = []
        for token in tokens:
            embeddings.append(self.model.get_vector(token))
        embeddings = np.array(embeddings)
        # For empty sequences, we use zeros with the dimensionality of the features on
        # the second dimension (this is the format expected by the CE codebase)
        if embeddings.size == 0:
            embeddings = np.zeros((0, self.dim))
        return embeddings

    #@typechecked
    #def set_device(self, device: torch.device):
    #    msg = f"w2v only supports CPU-based execution found {device.type}"
    #    assert device.type == "cpu", msg


class OpenAI_GPT(TextEmbedding):
    """This model produces 768-embeddings using a pretrained GPT model, introduced
    in the paper:

    Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018).
    Improving language understanding by generative pre-training,
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding
    _paper.pdf
    """

    def __init__(self):
        self.tokenizer = transformers.OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        model = transformers.OpenAIGPTModel.from_pretrained("openai-gpt")
        model.eval()
        super().__init__(model=model)

    @typechecked
    def text2vec(self, text: str) -> np.ndarray:
        tokenized_text = self.tokenizer.tokenize(text)

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = paddle.to_tensor(indexed_tokens, dtype='int64') #tokens_tensor = torch.LongTensor([indexed_tokens]).to(self.model.device)

        with paddle.no_grad():
            hidden_states = self.model(tokens_tensor)
            embeddings = hidden_states[0].numpy()
        return embeddings.squeeze(0)


