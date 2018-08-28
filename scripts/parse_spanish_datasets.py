import numpy as np

import re
import os


def clean_data(root_path):
    spanish_text, english_text = [], []

    files = [".es".join(x.split(".es")[:-1]) for x in os.listdir(root_path) if x.endswith(".es")]

    pattern = re.compile(r"([!\"#\$%&\'\(\)\*\+,-\.\/:;\<\=\>\?\@\[\\\]\^_`\{\|\}~])")
    empty_lines = re.compile(r"^\s*$")

    for f in files:
        spanish_file = open(root_path + f + ".es", 'r', encoding='utf8')
        english_file = open(root_path + f + ".en", 'r', encoding='utf8')

        while True:
            spanish_line = spanish_file.readline()
            english_line = english_file.readline()

            if not spanish_line or not english_line:
                break

            spanish_line = re.sub(pattern, r" \1 ", spanish_line.strip())
            spanish_line = re.sub(r"\s{2,}", " ", spanish_line)

            english_line = re.sub(pattern, r" \1 ", english_line.strip())
            english_line = re.sub(r"\s{2,}", " ", english_line)

            if not empty_lines.match(english_line) or not empty_lines.match(spanish_line):
                english_text.append(english_line)
                spanish_text.append(spanish_line)
        english_file.close()
        spanish_file.close()

    return spanish_text, english_text


def split_sets(spanish_set, english_set, val_percent, test_percent):
    assert len(spanish_set) == len(english_set)

    size = len(spanish_set)
    val_size = int(val_percent * size)
    test_size = int(test_percent * size)
    s = np.random.permutation(range(size))

    es_test, en_test = zip(*[(spanish_set[idx], english_set[idx]) for idx in s[:test_size]])
    es_val, en_val = zip(*[(spanish_set[idx], english_set[idx]) for idx in s[test_size: (test_size + val_size)]])
    es_train, en_train = zip(*[(spanish_set[idx], english_set[idx]) for idx in s[(test_size + val_size):]])

    return (es_train, en_train), (es_val, en_val), (es_test, en_test)


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
    path = os.path.dirname(os.path.abspath(__file__)) + "/../es-en-data/"
    es_corpus, en_corpus = [], []

    # clean data
    es, en = clean_data(root_path=path)

    # split sets out randomly
    train_sets, val_sets, test_sets = split_sets(spanish_set=es, english_set=en, val_percent=0.001, test_percent=0.001)

    # save consumable files
    write_to_text_files(train=train_sets, val=val_sets, test=test_sets, save_path="../data/es-en/")
