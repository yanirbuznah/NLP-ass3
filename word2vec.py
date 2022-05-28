import sys
import numpy as np


def load_vectors(file_name):
    words = []
    vecs = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        word, vector = line.split(maxsplit=1)
        words.append(word)
        vector = np.fromstring(vector.strip(), sep=' ')
        vecs.append(vector)
    return np.array(words), np.array(vecs)

def write_to_file_most_similars(similar_bow_5, similar_dep, words_to_test, output_file_name):
    with open(output_file_name, "w") as f:
        for i, word in enumerate(words_to_test):
            f.write(word + "\n")
            for x, y in zip(similar_bow_5[i], similar_dep[i]):
                f.write(x + " " + y + "\n")
            f.write("*" * 9 + "\n")


def calc_similar(words, W, w2i, test_words, k=20):
    # normalize row-wise
    W_norm = W / np.linalg.norm(W, axis=1)[:, None]
    sims = [] 
    for word in test_words:
        word_vec = W_norm[w2i[word]]
        sims_ids = W_norm.dot(word_vec).argsort()[-k - 1: -1][::-1]
        sims.append(words[sims_ids])
    return sims

def calc_k_higest_contexts(words, W, C, w2i, test_words, k=10):
    highest_contexts = []
    C_T = C.T
    for word in test_words:
        WC = np.matmul(W[w2i[word]], C_T)
        highest_ids = WC.argsort()[-k - 1: -1][::-1]
        highest_contexts.append(words[highest_ids])
    return highest_contexts


def main():
    dependency_based_words_file_name = sys.argv[1]
    bag_of_words_5_words_file_name = sys.argv[2]
    dependency_based_context_file_name = sys.argv[3]
    bag_of_words_5_context_file_name = sys.argv[4]

    test_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']

    # load word vectors
    dependency_based_words = load_vectors(dependency_based_words_file_name) # words, W
    bag_of_words_5_words = load_vectors(bag_of_words_5_words_file_name) # words, W

    w2i_dep_words = {w:i for i,w in enumerate(dependency_based_words[0])}
    w2i_bow_5_words = {w:i for i,w in enumerate(bag_of_words_5_words[0])}

    # # calc similarities between words
    similar_bow_5 = calc_similar(dependency_based_words[0], dependency_based_words[1], w2i_dep_words, test_words)
    similar_dep = calc_similar(bag_of_words_5_words[0], bag_of_words_5_words[1], w2i_bow_5_words, test_words)
    
    write_to_file_most_similars(similar_bow_5, similar_dep, test_words, "top20_w2v.txt")

    # load context vectors
    dependency_based_context = load_vectors(dependency_based_context_file_name) # words, W
    bag_of_words_5_context = load_vectors(bag_of_words_5_context_file_name) # words, W
    
    dep_highest_contexts = calc_k_higest_contexts(dependency_based_context[0], dependency_based_words[1], dependency_based_context[1], w2i_dep_words, test_words)
    bow_5_highest_contexts = calc_k_higest_contexts(bag_of_words_5_context[0], bag_of_words_5_words[1], bag_of_words_5_context[1], w2i_bow_5_words, test_words)

    write_to_file_most_similars(bow_5_highest_contexts, dep_highest_contexts, test_words, "top10_contexts_w2v.txt")


if __name__ == "__main__":
    main()