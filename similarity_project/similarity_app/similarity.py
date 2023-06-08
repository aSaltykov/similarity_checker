import os

from keras.models import load_model
from keras.utils import custom_object_scope
from transformers.models.bert.modeling_tf_bert import TFBertMainLayer

from .module_sim import SimilarityChecker, \
    EncodingConfig

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'final_model.h5')
with custom_object_scope({'TFBertMainLayer': TFBertMainLayer}):
    model = load_model(model_path)

similarity_checker = SimilarityChecker(model, EncodingConfig(max_len=128))
