import codecs
import regex
from bert.extract_feature import BertVector
from sklearn.metrics.pairwise import euclidean_distances


def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('/home/lidong/tangle/muti_model/multi-data/de.vocab.tsv', 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab) if idx > 3}
    idx2word = {idx: word for idx, word in enumerate(vocab) if idx > 3}
    return word2idx, idx2word


def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('/home/lidong/tangle/muti_model/multi-data/en.vocab.tsv', 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab) if idx > 3}
    idx2word = {idx: word for idx, word in enumerate(vocab) if idx > 3}
    return word2idx, idx2word


def load_vec_dic(idx: dict, bert: BertVector):
    temp_dic = {key : [] for key in idx.keys()}
    for key in temp_dic:
        temp_dic[key] = bert.encode([key])
    return temp_dic


def find_nearby(vec_dic: dict, bert: BertVector, word: str):
    if word in vec_dic:
        return word
    word_vec = bert.encode([word])
    temp_dic = {}
    for key in vec_dic:
        temp_dic[key] = euclidean_distances(vec_dic[key], word_vec)[0][0]
    return min(temp_dic, key=temp_dic.get)


def main():
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    bert = BertVector()
    de_vec_dic = load_vec_dic(de2idx, bert)
    en_vec_dic = load_vec_dic(en2idx, bert)
    with open("/home/lidong/tangle/muti_model/multi-data/val.de", 'r', encoding="utf-8") as f_in, open("/home/lidong/tangle/muti_model/multi-data/val_rep.de", 'w', encoding="utf-8") as f_out:
        for line in f_in:
            line = regex.sub("[^\s\p{Latin}']", "", line)
            data = line.strip().split(" ")
            data = [x for x in data if x != ""]
            temp_list = []
            for x in data:
                temp_list.append(find_nearby(de_vec_dic, bert, x))
            f_out.write(" ".join(temp_list) + "\n")

    with open("/home/lidong/tangle/muti_model/multi-data/val.en", 'r', encoding="utf-8") as f_in, open("/home/lidong/tangle/muti_model/multi-data/val_rep.en", 'w', encoding="utf-8") as f_out:
        for line in f_in:
            line = regex.sub("[^\s\p{Latin}']", "", line)
            data = line.strip().split(" ")
            data = [x for x in data if x != ""]
            temp_list = []
            for x in data:
                temp_list.append(find_nearby(en_vec_dic, bert, x))
            f_out.write(" ".join(temp_list) + "\n")


if __name__ == '__main__':
    main()
