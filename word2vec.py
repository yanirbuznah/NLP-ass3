import sys
import numpy as np


def load_and_norm_vectors(file_name):
    words = []
    vecs = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        word, vector = line.split(maxsplit=1)
        words.append(word)
        vector = np.fromstring(vector.strip(), sep=' ')
        vector /= np.linalg.norm(vector)
        vecs.append(vector)
    return  np.array(words), np.array(vecs)

def write_to_file_most_similars(similar_bow_5, similar_dep, similar_bow_5_contexts, words_to_test, output_file_name):
    with open(output_file_name, "w") as f:
        for i, word in enumerate(words_to_test):
            f.write(word + "\n")
            for x, y, z in zip(similar_bow_5[i], similar_dep[i], similar_bow_5_contexts[i]):
                f.write(x + " " + y + " " + z + "\n")
            f.write("*" * 9 + "\n")


def calc_similar(file_name, test_words, k=20):
    sims = [] 
    words, W = load_and_norm_vectors(file_name)
    w2i = {w:i for i,w in enumerate(words)}
    for word in test_words:
        word_vec = W[w2i[word]]
        sims_ids = W.dot(word_vec).argsort()[-1:-k:-1]
        sims.append(words[sims_ids])
    return sims

def main():
    dependency_based_words_file_name = sys.argv[1]
    dependency_based_context_file_name = sys.argv[2]
    bag_of_words_5_words_file_name = sys.argv[3]
    bag_of_words_5_context_file_name = sys.argv[4]

    # dependency_files = [dependency_based_words_file_name, dependency_based_context_file_name]
    # bag_of_words_files = [bag_of_words_5_words_file_name, bag_of_words_5_context_file_name]

    test_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']

    similar_bow_5 = calc_similar(bag_of_words_5_words_file_name, test_words)
    similar_dep = calc_similar(dependency_based_words_file_name, test_words)

    similar_bow_5_contexts = calc_similar(bag_of_words_5_context_file_name, test_words)
    # similar_dep_contexts = calc_similar(dependency_based_context_file_name, test_words)

    write_to_file_most_similars(similar_bow_5, similar_dep, similar_bow_5_contexts, test_words, "top20_w2v_contexts.txt")
    # write_to_file_most_similars(similar_bow_5_contexts, similar_dep_contexts, test_words, "top20_w2v_contexts.txt")






if __name__ == "__main__":
    main()