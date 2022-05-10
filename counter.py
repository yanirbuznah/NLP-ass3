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


def count_words(sentences):
    words = {}
    for sentence in sentences:
        for word in sentence:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    return words


def write_count_to_file(file_name, words_counts, k=50):
    words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    with open(file_name, 'w', encoding='utf-8') as f:
        for word, count in words[:k]:
            f.write(word + ' ' + str(count) + '\n')


def context_counter(sentences):
    counts = defaultdict(Counter)
    for s in sentences:
        for w in s:
            counts[w] = Counter() if w not in counts else counts[w]
            for w2 in s:
                if w != w2:
                    counts[w][w2] = 1 if w2 not in counts[w] else counts[w][w2] + 1

    return counts

def main():
    file_name = sys.argv[1]
    # read lemma sentences from file
    sentences = read_file(file_name)

    # read lemma sentences from file
    # sentences = read_table_from_file(file_name)

    # count words
    words = count_words(sentences)

    # write words to file
    write_count_to_file('words.txt', words)

    # count context
    context = context_counter(sentences)

    # write context to file
    context_list = {k:sum(v.values()) for k,v in context.items()}
    write_count_to_file('context.txt', context_list)

if __name__ == '__main__':
    main()
