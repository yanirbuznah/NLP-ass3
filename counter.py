import sys
from collections import defaultdict, Counter

import pandas as pd


def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sentences = []
    line = []
    for l in lines:

        if l == '\n':
            sentences.append(line)
            line = []
        else:

            line.append(l.split('\t')[2])
    return sentences


def read_table_from_file(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None)
    df.columns = ['ID', 'FORM', 'LEMMA', 'COARSE-POS', 'FINE-POS', '-', 'HEAD', 'DEP-TYPE', '-', '-']
    print(df.shape)
    print(df.head())
    return df


def count_words(sentences):
    words = {}
    idx_to_word = {}
    word_to_idx = {}

    for sentence in sentences:
        for word in sentence:
            idx = get_idx_from_dicts(idx_to_word, word_to_idx, word)
            if idx in words:
                words[idx] += 1
            else:
                words[idx] = 1
    return words, idx_to_word, word_to_idx


def write_count_to_file(file_name, words_counts, idx_to_word, k=50):
    words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    with open(file_name, 'w', encoding='utf-8') as f:
        for idx, count in words[:k]:
            f.write(idx_to_word[idx] + ' ' + str(count) + '\n')


def get_idx_from_dicts(idx_to_word, word_to_idx, word):
    if word not in word_to_idx.keys():
        word_to_idx[word] = len(word_to_idx)
        idx_to_word[len(idx_to_word)] = word
    return word_to_idx[word]


def context_counter(sentences, window=2):
    counts = defaultdict(Counter)
    idx_to_word = {}
    word_to_idx = {}
    for s in sentences:
        for i, w in enumerate(s):
            idx = get_idx_from_dicts(idx_to_word, word_to_idx, w)
            if idx not in counts.keys():
                counts[idx] = Counter()
            count_context_in_range(counts, word_to_idx, idx_to_word, s, i, window)

    return counts, idx_to_word, word_to_idx


def count_context_in_range(counts, word_to_idx, idx_to_word, s, index, window=2):
    start = max(0, index - window)
    end = min(len(s), index + window + 1)

    word_idx = word_to_idx[s[index]]
    for j in range(start, index):
        context_idx = get_idx_from_dicts(idx_to_word, word_to_idx, s[j])
        counts[word_idx][context_idx] = 1 if context_idx not in counts[word_idx] else counts[word_idx][context_idx] + 1

    for j in range(index + 1, end):
        context_idx = get_idx_from_dicts(idx_to_word, word_to_idx, s[j])
        counts[word_idx][context_idx] = 1 if context_idx not in counts[word_idx] else counts[word_idx][context_idx] + 1


def pandas_tries(file_name):
    df = read_table_from_file(file_name)
    counts = df['LEMMA'].value_counts()
    starts_indexes = df.index[df['ID'] == 1].tolist() + [len(df)]
    list_of_sentences = [df.iloc[starts_indexes[n]:starts_indexes[n + 1]] for n in range(len(starts_indexes) - 1)]





def main():
    file_name = sys.argv[1]
    # read lemma sentences from file
    sentences = read_file(file_name)

    # # read lemma sentences from file
    pandas_tries(file_name)
    # df = read_table_from_file(file_name)
    # counts = df['LEMMA'].value_counts()
    # starts_indexes = df.index[df['ID'] == 1].tolist() + [len(df)]
    # list_of_sentences = [df.iloc[starts_indexes[n]:starts_indexes[n+1]] for n in range(len(starts_indexes)-1)]

    # count words
    words, idx_to_word, word_to_idx = count_words(sentences)

    # write words to file
    write_count_to_file('words.txt', words, idx_to_word)

    # count context
    context, idx_to_word, word_to_idx = context_counter(sentences, window=sys.maxsize)

    # write context to file
    context_list = {k: sum(v.values()) for k, v in context.items()}
    write_count_to_file('context_max_window.txt', context_list, idx_to_word)

    # count context
    context, idx_to_word, word_to_idx = context_counter(sentences, window=2)

    # write context to file
    context_list = {k: sum(v.values()) for k, v in context.items()}
    write_count_to_file('context_window2.txt', context_list, idx_to_word)


if __name__ == '__main__':
    main()
