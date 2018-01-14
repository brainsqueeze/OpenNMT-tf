import numpy as np
from nltk.tokenize import word_tokenize

import os

MAX_SEQ_LEN = 30


def parse_wiki_data(file_path):
    hindi_text, english_text = [], []

    with open(file_path, "r", encoding="utf8") as f:
        while True:
            line = f.readline()

            if not line:
                break

            hindi, english = [text.strip() for text in line.split("|||")]

            if len(word_tokenize(english)) <= MAX_SEQ_LEN:
                hindi_text.append(hindi)
                english_text.append(english)
    return hindi_text, english_text


def parse_hind_en_corp(file_path):
    hindi_text, english_text = [], []

    with open(file_path, "r", encoding="utf8") as f:
        while True:
            line = f.readline()

            if not line:
                break
            source, _, label, english, hindi = line.split("\t")

            if len(word_tokenize(english)) <= MAX_SEQ_LEN:
                hindi_text.append(hindi.strip())
                english_text.append(english.strip())

    return hindi_text, english_text


def parse_separate_training_files(hi_file_path, en_file_path):
    hindi_text, english_text = [], []

    hindi_corpus = open(hi_file_path, "r", encoding="utf8")
    english_corpus = open(en_file_path, "r", encoding="utf8")

    while True:
        hi_line = hindi_corpus.readline()
        en_line = english_corpus.readline()

        if not hi_line and not en_line:
            break

        if len(word_tokenize(en_line)) <= MAX_SEQ_LEN:
            hindi_text.append(hi_line.strip())
            english_text.append(en_line.strip())

    hindi_corpus.close()
    english_corpus.close()

    return hindi_text, english_text


def split_sets(hindi_set, english_set, val_percent, test_percent):
    assert len(hindi_set) == len(english_set)

    size = len(hindi_set)
    val_size = int(val_percent * size)
    test_size = int(test_percent * size)
    s = np.random.permutation(range(size))

    hi_test, en_test = zip(*[(hindi_set[idx], english_set[idx]) for idx in s[:test_size]])
    hi_val, en_val = zip(*[(hindi_set[idx], english_set[idx]) for idx in s[test_size: (test_size + val_size)]])
    hi_train, en_train = zip(*[(hindi_set[idx], english_set[idx]) for idx in s[(test_size + val_size):]])

    return (hi_train, en_train), (hi_val, en_val), (hi_test, en_test)


def write_to_text_files(train, val, test, save_path):
    with open(save_path + "src-train.txt", "w") as f:
        for row in train[0]:
            f.write(row + "\n")

    with open(save_path + "tgt-train.txt", "w") as f:
        for row in train[1]:
            f.write(row + "\n")

    with open(save_path + "src-val.txt", "w") as f:
        for row in val[0]:
            f.write(row + "\n")

    with open(save_path + "tgt-val.txt", "w") as f:
        for row in val[1]:
            f.write(row + "\n")

    with open(save_path + "src-test.txt", "w") as f:
        for row in test[0]:
            f.write(row + "\n")

    with open(save_path + "tgt-test.txt", "w") as f:
        for row in test[1]:
            f.write(row + "\n")
    return


if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__)) + "/../hindi-en-data/"
    hi_corpus, en_corpus = [], []

    hi, en = parse_wiki_data(file_path=path + "wiki/hi-en/wiki-titles.hi-en")
    hi_corpus.extend(hi)
    en_corpus.extend(en)

    hi, en = parse_hind_en_corp(file_path=path + "hindencorp05.plaintext")
    hi_corpus.extend(hi)
    en_corpus.extend(en)

    hi, en = parse_separate_training_files(hi_file_path=path + "training.hi-en.hi",
                                           en_file_path=path + "training.hi-en.en")
    hi_corpus.extend(hi)
    en_corpus.extend(en)

    train_sets, val_sets, test_sets = split_sets(hindi_set=hi_corpus,
                                                 english_set=en_corpus,
                                                 val_percent=0.05,
                                                 test_percent=0.05)

    write_to_text_files(train=train_sets,
                        val=val_sets,
                        test=test_sets,
                        save_path="../data/hi-en/")
