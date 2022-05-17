import sys
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

FUNCTION_WORDS = {'to', 'the', 'of', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'she'}


def read_table_from_file(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)
    df.columns = ['ID', 'FORM', 'LEMMA', 'COARSE-POS', 'FINE-POS', '-', 'HEAD', 'DEP-TYPE', '-', '-']
    # df.drop(set(df.columns) - {'ID', 'FORM', 'LEMMA', 'HEAD'}, axis=1, inplace=True)
    df.drop(['COARSE-POS', 'FINE-POS', '-', '-', '-'], axis=1, inplace=True)
    print(df.shape)
    print(df.head())
    return df


def count_context_by_dep(counts, word_to_idx, s, index):
    word_idx = word_to_idx[s['LEMMA'][index]]

    dep_type = s['DEP-TYPE'][index]

    #TODO: check if this is correct behavior
    if dep_type == 'ROOT':
        return

    head = s['LEMMA'][s['HEAD'][index]]
    head_idx = word_to_idx[head]

    counts[word_idx][(head_idx, 1, dep_type)] += 1
    counts[head_idx][(word_idx, -1, dep_type)] += 1




def context_counter(sentences, words_to_idx, idx_to_words, task='full_window'):
    counts = defaultdict(Counter)
    for s in sentences:
        s.index = np.arange(1, len(s) + 1)
        for i, w in enumerate(s['LEMMA']):

            if task == 'full_window':
                count_context_in_range(counts, words_to_idx, s['LEMMA'], i + 1, sys.maxsize)
            elif task == '2_window':
                count_context_in_range(counts, words_to_idx, s['LEMMA'], i + 1, 2)
            else:
                count_context_by_dep(counts, words_to_idx, s, i + 1)

    return counts


def count_context_in_range(counts, word_to_idx, s, index, window=2):
    start = max(1, index - window)
    end = min(len(s) + 1, index + window)
    # s = s.LEMMA
    word_idx = word_to_idx[s[index]]
    j = index - 1
    while j >= start and j > 0:
        if s[j] in FUNCTION_WORDS:
            start -= 1
        else:
            context_idx = word_to_idx[s[j]]
            counts[word_idx][context_idx] += 1
        j -= 1

    j = index + 1
    while j <= end and j <= len(s):
        if s[j] in FUNCTION_WORDS:
            end += 1
        else:
            context_idx = word_to_idx[s[j]]
            counts[word_idx][context_idx] += 1
        j += 1




def main():
    file_name = sys.argv[1]
    df = read_table_from_file(file_name)
    counts = df['LEMMA'].value_counts()
    starts_indexes = df.index[df['ID'] == 1].tolist() + [len(df)]
    list_of_sentences = [df.iloc[starts_indexes[n]:starts_indexes[n + 1]] for n in range(len(starts_indexes) - 1)]
    idx_to_words = df.LEMMA.unique()
    words_to_idx = {word: idx for idx, word in enumerate(idx_to_words)}

    counts = context_counter(list_of_sentences, words_to_idx, idx_to_words, task='full_window')
    context_list = {k: sum(v.values()) for k, v in counts.items()}
    context_list = sorted(context_list.items(), key=lambda x: x[1], reverse=True)

    counts = context_counter(list_of_sentences, words_to_idx, idx_to_words, task='2_window')
    context_list2 = {k: sum(v.values()) for k, v in counts.items()}
    context_list2 = sorted(context_list2.items(), key=lambda x: x[1], reverse=True)

    counts = context_counter(list_of_sentences, words_to_idx, idx_to_words, task='dep_type')
    # context_list3 = {k: sum(v.values()) for k, v in counts.items()}
    # context_list3 = sorted(context_list3.items(), key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    main()
