import pickle
import sys

import numpy as np
import torch
from natasha import Segmenter, Doc
from torch import nn
from tqdm import tqdm
sys.path.append('.\\data')
sys.path.append('.\\db_utils')
from constants import DB_PATH, TABLE_NAME
from read_vacs_by_city import read_column_by_city
from model import create_model



def load_vocabs():
    '''
    Очень тупая функция, но я устал
    :return:
    '''
    with open('.\\data\\res_vocab.pickle', 'rb') as f:
        res_vocab = pickle.load(f)

    with open('.\\data\\vac_vocab.pickle', 'rb') as f:
        vac_vocab = pickle.load(f)

    return vac_vocab, res_vocab


def make_indexes_from_tuple(t, vocab):
    '''
    Предобработка текста перед подачей в модель
    '''
    segmenter = Segmenter()

    indexes = []
    for sent in t:
        sent = Doc(sent)
        sent.segment(segmenter)
        ind = np.array([vocab.get(token.text.lower()) for token in sent.tokens], dtype=np.float16)
        if len(ind) == 0:
            if len(vocab) == 9425:
                # ищу любую работу
                ind = np.array([8100, 7644, 6224])
            else:
                # ищу лучшего работника
                ind = np.array([16400, 8216, 15424])

        ind = np.nan_to_num(ind, nan=0)
        ind = torch.LongTensor(ind)

        indexes.append(ind)

    padded_indexes = nn.utils.rnn.pad_sequence(indexes, padding_value=0, batch_first=True)
    return padded_indexes


def model_inference_one_vac(model, vac_vocab, res_vocab, vac_text, city, device):
    '''
    Инференс на 1 вакансии
    '''

    result = []

    # проваливаемся в кластер по локации
    cluster = read_column_by_city(DB_PATH, TABLE_NAME, 'res_des', city)

    vac_idx = make_indexes_from_tuple(tuple([vac_text]), vac_vocab).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for res_text in tqdm(cluster):
            if res_text:
                res_idx = make_indexes_from_tuple(tuple([res_text]), res_vocab).to(device)
                similarity = model(vac_idx, res_idx).item()
            else:
                similarity = -1

            result.append(tuple([res_text, similarity]))

    result.sort(key=lambda x: -x[1])

    return result


if __name__ == "__main__":
    vac_vocab, res_vocab = load_vocabs()

    model = create_model()
    final_state = torch.load('.\\data\\final_model_state.pth', map_location='cuda')
    model.load_state_dict(final_state)

    # это пример
    test_text = 'ищу высокоточного специалиста на супер сложную работу в предприятие по производству меда'
    test_city = 'чита'

    result = model_inference_one_vac(model, vac_vocab, res_vocab, test_text, test_city, 'cuda')

    print(result)
