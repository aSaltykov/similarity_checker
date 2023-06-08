from dataclasses import dataclass
from random import Random

import numpy as np
from keras.utils import Sequence as MySequence
from transformers import BertTokenizer

do_token = BertTokenizer.from_pretrained


@dataclass
class EncodingConfig:
    max_len: int
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = True
    padding: str = 'max_length'
    return_tensors: str = "tf"
    truncation: bool = True

    def get_params(self):
        params = self.__dict__.copy()
        params.pop("max_len")
        params["max_length"] = self.max_len
        return params


class BertEncoder:
    def __init__(self):
        self.token_text = do_token(
            "bert-base-multilingual-cased", do_lower_case=False
        )

    def transform_to_bert(self, input_texts, cfg):
        transformed = self.token_text.batch_encode_plus(
            [list(text) for text in input_texts], **cfg.get_params()
        )
        return transformed


class GenText:
    def __init__(self, text_data, group):
        self.text_data = text_data
        self.group = group
        self.on_init()

    def on_init(self):
        self.idxs = np.arange(len(self.text_data))

    def generate_groups(self, idx):
        slice_obj = slice(idx * self.group, (idx + 1) * self.group)
        idxs_list = self.idxs[slice_obj].tolist()
        result = []
        for i in idxs_list:
            result.append(self.text_data[i])
        return result


class TextTransformer:
    def __init__(self, modified):
        self.modified = modified

    def transform_text(self, text_data, cfg):
        data = self.modified.transform_to_bert(text_data, cfg)
        result = []
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            result.append(np.array(data[key], dtype="int32"))
        return result


class EncodedDataGenerator:
    def __init__(self, data, group, cfg):
        self.data = GenText(data, group)
        self.data_transformer = TextTransformer(BertEncoder())
        self.cfg = cfg

    def generate_encoded_groups(self, idx):
        data_in = self.data.generate_groups(idx)
        transformed_data = self.data_transformer.transform_text(data_in, self.cfg)
        return transformed_data


class DataPreparer(MySequence):
    def __init__(self, data_in, marks, group, cfg=None):
        self.changed_data = EncodedDataGenerator(data_in, group, cfg)
        self.marks = marks

    @property
    def text_length(self):
        data_in = self.changed_data.data.text_data
        length = len(data_in)
        return length

    @property
    def group(self):
        return self.changed_data.data.group

    def __len__(self):
        total_texts = self.text_length
        group = self.group
        num_groups, _ = divmod(total_texts, group)
        return num_groups

    def __getitem__(self, idx):
        changed_data = self.changed_data.generate_encoded_groups(idx)
        if self.marks is not None:
            group_idxs = self.changed_data.data.idxs[idx * self.group: (idx + 1) * self.group]
            marks = np.array([self.marks[j] for j in group_idxs], dtype="int32").reshape(-1, 1)
            return changed_data, marks
        else:
            return changed_data


class RandomDataMixer:
    def __init__(self, total_size, random_seed=42):
        self.index_list = [i for i in range(total_size)]
        self.random_generator = Random(random_seed)

    def shuffle_indices(self):
        self.random_generator.shuffle(self.index_list)


class BertSimilarityDataGenerator(DataPreparer):
    def __init__(self, input_texts, marks, group, mix=True, cfg=None):
        super().__init__(input_texts, marks, group, cfg)
        self.mix = mix
        self.mixer = RandomDataMixer(len(input_texts))
        self.mix_index()

    def mix_index(self):
        if self.mix:
            self.perform_mix()

    def perform_mix(self):
        self.mixer.shuffle_indices()


class SimilarityChecker:
    def __init__(self, model, encoding_config):
        self.model = model
        self.encoding_config = encoding_config

    def _format_probability(self, probabilities, index):
        percentage = probabilities[index] * 100
        return f"{percentage: .2f}%"

    def _get_prediction(self, probabilities, index):
        if probabilities[index] >= 0.6:
            return 'Similar'
        else:
            return 'Not similar'

    def check_similarity(self, text1, text2):
        text_pairs = [[str(text1), str(text2)]]
        test_data = BertSimilarityDataGenerator(
            text_pairs, marks=None, group=1, mix=False, cfg=self.encoding_config
        )
        probabilities, *_ = self.model.predict(test_data[0])
        max_index = np.argmax(probabilities)
        formatted_probability = self._format_probability(probabilities, max_index)
        prediction = self._get_prediction(probabilities, max_index)

        return prediction, formatted_probability
