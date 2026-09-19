# coding=utf-8
"""CodeSage configuration class."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CodeSageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CodeSageModel`].
    It is used to instantiate a CodeSage model according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    model outputs. Read the documentation from [`PretrainedConfig`] for more details.

    Args:
        vocab_size (`int`, *optional*, defaults to 50000):
            Vocabulary size of the CodeSage model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
        residual_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        embedding_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embedding layers.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        padding_idx (`int`, *optional*, defaults to 1):
            Padding token id.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the classification head.
        problem_type (`str`, *optional*):
            Problem type for `CodeSageForSequenceClassification`. Can be "regression",
            "single_label_classification", or "multi_label_classification".
    """

    model_type = "codesage"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        activation_function: str = "gelu",
        residual_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        embedding_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        padding_idx: int = 1,
        classifier_dropout: float = 0.1,
        problem_type: str = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=padding_idx,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.residual_dropout_prob = residual_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.padding_idx = padding_idx
        self.classifier_dropout = classifier_dropout
        self.problem_type = problem_type
