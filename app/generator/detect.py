import torch
from app.generator.model import Generator
from conf.modelconf.generator_conf import config


class GeneratorModel(object):
    def __init__(self, init=False):
        if init:
            load_model = "library/init_generator.model"
            model_state_name = "model_stat"
            dataset_name = "dataset_operator"
        else:
            load_model = "library/generator.model"
            model_state_name = "gen_model_stat"
            dataset_name = "gen_dataset"
        model_state = torch.load(load_model, map_location=config.device)
        self.dataset = model_state[dataset_name]
        self.vectorize = self.dataset.get_vectorizer()
        self.vocabulary = self.vectorize.get_vocabulary()
        self.model = Generator(embedding_size=config.embedding_size,
                               num_layers=config.num_layers,
                               hidden_size=config.hidden_size,
                               vectorize_size=len(self.vocabulary),
                               embedding_weight=None).to(config.device)
        self.model.load_state_dict(model_state[model_state_name])

    def detect(self, sentence=None, num=20):
        if sentence:
            sentence = sentence.split(" ")
            sentence_list = []
            for i in range(len(sentence)):
                sentence_list.append(self.vocabulary.look_up_token(sentence[i]))
        else:
            sentence_list = None

        result = self.vectorize.reverse_list(self.model.random_generate(sentence_list, num))
        print(result)
        return result


if __name__ == "__main__":
    test = GeneratorModel()
    test.detect()