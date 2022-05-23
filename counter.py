import sys
import time
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

# def read_file(file_name):
#     with open(file_name, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     sentences = []
#     line = []
#     for l in lines:
#
#         if l == '\n':
#             sentences.append(line)
#             line = []
#         else:
#
#             line.append(l.split('\t')[2])
#     return sentences
#
#
# def read_table_from_file(file_name):
#     df = pd.read_csv(file_name, sep='\t', header=None)
#     df.columns = ['ID', 'FORM', 'LEMMA', 'COARSE-POS', 'FINE-POS', '-', 'HEAD', 'DEP-TYPE', '-', '-']
#     print(df.shape)
#     print(df.head())
#     return df
#
#
# def count_words(sentences):
#     words = {}
#     idx_to_word = {}
#     word_to_idx = {}
#
#     for sentence in sentences:
#         for word in sentence:
#             idx = get_idx_from_dicts(idx_to_word, word_to_idx, word)
#             if idx in words:
#                 words[idx] += 1
#             else:
#                 words[idx] = 1
#     return words, idx_to_word, word_to_idx
#
#

#
#
# def get_idx_from_dicts(idx_to_word, word_to_idx, word):
#     if word not in word_to_idx.keys():
#         word_to_idx[word] = len(word_to_idx)
#         idx_to_word[len(idx_to_word)] = word
#     return word_to_idx[word]
#
#
# def context_counter(sentences, window=2):
#     counts = defaultdict(Counter)
#     idx_to_word = {}
#     word_to_idx = {}
#     for s in sentences:
#         for i, w in enumerate(s):
#             idx = get_idx_from_dicts(idx_to_word, word_to_idx, w)
#             if idx not in counts.keys():
#                 counts[idx] = Counter()
#             count_context_in_range(counts, word_to_idx, idx_to_word, s, i, window)
#
#     return counts, idx_to_word, word_to_idx
#
#
# def count_context_in_range(counts, word_to_idx, idx_to_word, s, index, window=2):
#     start = max(0, index - window)
#     end = min(len(s), index + window + 1)
#
#     word_idx = word_to_idx[s[index]]
#     for j in range(start, index):
#         context_idx = get_idx_from_dicts(idx_to_word, word_to_idx, s[j])
#         counts[word_idx][context_idx] = 1 if context_idx not in counts[word_idx] else counts[word_idx][context_idx] + 1
#
#     for j in range(index + 1, end):
#         context_idx = get_idx_from_dicts(idx_to_word, word_to_idx, s[j])
#         counts[word_idx][context_idx] = 1 if context_idx not in counts[word_idx] else counts[word_idx][context_idx] + 1
#

# 1235 - 9
# find the IN word - in this case 5
# check the head of the IN word - in this case 4
# if its the target word then find another word which head is the preposition (IN) - in this case 7
# save a context for the target word (4): (7, adpmod-for,1/-1)

FUNCTION_WORDS = {
    "to",
    "the",
    "of",
    "and",
    "a",
    "in",
    "is",
    "it",
    "you",
    "that",
    "he",
    "she",
}

def write_features_count_to_file(file_name, words_counts, idx_to_word, k=50):
    words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    directions = {1:"U", -1:"D"}
    with open(file_name, "w", encoding="utf-8") as f:
        for (idx,direction,dep_type), count in words[:k]:
            f.write(f"{idx_to_word[idx]}_{directions[direction]}_{dep_type} {str(count)}\n")

def write_count_to_file(file_name, words_counts, idx_to_word, k=50):
    words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
    with open(file_name, "w", encoding="utf-8") as f:
        for idx, count in words[:k]:
            f.write(f"{idx_to_word[idx]} {str(count)}\n")

def read_table_from_file(file_name):
    df = pd.read_csv(file_name, sep="\t", header=None, na_filter=False)
    df.columns = [
        "ID",
        "FORM",
        "LEMMA",
        "COARSE-POS",
        "FINE-POS",
        "-",
        "HEAD",
        "DEP-TYPE",
        "-",
        "-",
    ]
    # df.drop(set(df.columns) - {'ID', 'FORM', 'LEMMA', 'HEAD'}, axis=1, inplace=True)
    df.drop(["COARSE-POS", "-", "-", "-"], axis=1, inplace=True)
    print(df.shape)
    print(df.head())
    return df

def count_context_by_dep(counts, context, word_to_idx, s, index):
    word_idx = word_to_idx[s["LEMMA"][index]]
    dep_type = s["DEP-TYPE"][index]

    # TODO: check if this is correct behavior
    if dep_type == "ROOT":
        return

    head = s["LEMMA"][s["HEAD"][index]]
    if head not in word_to_idx:
        return
    head_idx = word_to_idx[head]

    # find the row where the head is the index
    if s["FINE-POS"][index] == "IN":
        prep_is_head = s[s["HEAD"] == index]
        if len(prep_is_head) == 0:
            return
        context[(head_idx, 1, (dep_type, s["LEMMA"][index]))] += len(prep_is_head)
        for i, row in prep_is_head.iterrows():
            if row["LEMMA"] not in word_to_idx:
                continue
            prep_is_head_idx = word_to_idx[row["LEMMA"]]
            counts[prep_is_head_idx][(head_idx, 1, (dep_type, s["LEMMA"][index]))] += 1

        # TODO: check what to do if there is more than one prep_is_head
        prep_is_head_idx = prep_is_head[
            prep_is_head["FINE-POS"].isin(["NN", "NNP", "NNS"])
        ]
        if len(prep_is_head_idx) == 0:
            return
        if prep_is_head_idx.iloc[0]["LEMMA"] not in word_to_idx:
            return
        prep_is_head_idx = word_to_idx[prep_is_head_idx.iloc[0]["LEMMA"]]
        counts[head_idx][(prep_is_head_idx, -1, (dep_type, s["LEMMA"][index]))] += 1
        context[(prep_is_head_idx, -1, (dep_type, s["LEMMA"][index]))] += 1
    else:
        counts[word_idx][(head_idx, 1, dep_type)] += 1
        counts[head_idx][(word_idx, -1, dep_type)] += 1
        context[(head_idx, 1, dep_type)] += 1
        context[(word_idx, -1, dep_type)] += 1

def context_counter(sentences, words, words_to_idx, task="full_window"):
    counts = defaultdict(Counter)
    context = Counter()
    for s in sentences:
        # used for dep contexts
        s.index = np.arange(1, len(s) + 1)
        for i, w in enumerate(s["LEMMA"]):
            # only calculate contexts for common words
            if w not in words_to_idx or words_to_idx[w] not in words.keys():
                continue

            if task == "full_window":
                count_context_in_range(counts, context, words_to_idx, s["LEMMA"], i + 1, sys.maxsize)
            elif task == "2_window":
                count_context_in_range(counts, context, words_to_idx, s["LEMMA"], i + 1, 2, ignore_function_words=True)
            else:
                count_context_by_dep(counts, context, words_to_idx, s, i + 1)
    return counts, context

def count_context_in_range(counts, context, word_to_idx, s, index, window=2, ignore_function_words=False):
    start = max(1, index - window)
    end = min(len(s) + 1, index + window)

    word_idx = word_to_idx[s[index]]
    j = index - 1
    while j >= start and j > 0:
        if ignore_function_words and s[j] in FUNCTION_WORDS:
            start -= 1
        elif s[j] in word_to_idx:
            context_idx = word_to_idx[s[j]]
            counts[word_idx][context_idx] += 1
            context[context_idx] += 1
        j -= 1

    j = index + 1

    while j <= end and j <= len(s):
        if ignore_function_words and s[j] in FUNCTION_WORDS:
            end += 1
        elif s[j] in word_to_idx:
            context_idx = word_to_idx[s[j]]
            counts[word_idx][context_idx] += 1
            context[context_idx] += 1
        j += 1

def vectorize_counts(context_counts, words, context_list, word_to_idx, threshold=100, dep=False):
    vectors = []
    # iterate over all words
    for idx in words.keys():
        # counts = combine_dep_contexts(context_counts) if dep else context_counts
        # sort the context counts by frequency
        counts = sorted(context_counts[idx].items(), key=lambda x: x[1], reverse=True)
        # create a vector of zeros and trim the counts to the threshold (100)
        counts = counts[:threshold]
        # vector = np.zeros(len(words))
        vector = np.zeros(len(word_to_idx))
        # f = filter(lambda x: x[0] in words.keys(), counts) if not dep else filter(lambda x: x[0][0] in words.keys(), counts)
        # f = filter(lambda x: x[0] in context_list.keys(), counts)
        # iterate over the counts and set the corresponding indices to the frequency
        for (context_idx, count) in counts:
            if dep:
                context_idx = context_idx[0]
            vector[context_idx] += count #if context_idx in context_list else 0
        vectors.append(vector)
    vectors = np.array(vectors)
    return vectors

def pmi_from_matrix(matrix):
    pmi_matrix = np.zeros(matrix.shape)
    log_matrix_sum = np.log(matrix.sum())
    sparse_pmi_matrix = dict()
    fliped_pmi_matrix = dict()
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            if col > 0:
                """
                pmi = log(p(x,y) / (p(x) * p(y))) =
                p(x,y) = matrix[i,j] / matrix.sum(), p(x) = row_sums[i] / matrix.sum() , p(y) = col_sums[j] / matrix.sum()
                pmi = log(p(x,y) / (p(x) * p(y))) = log(matrix[i,j] / matrix.sum()) / (row_sums[i] / matrix.sum()) * (col_sums[j] / matrix.sum())
                = log(matrix[i,j] / matrix.sum()) / (row_sums[i] * col_sums[j]/ matrix.sum()^2)) = log((matrix[i,j] / row_sums[i] * col_sums[j]) * matrix.sum())
                = log(matrix[i,j] / row_sums[i] * col_sums[j]) + log(matrix.sum())
                """
                p_xy = matrix[i, j]
                p_x = row_sums[i]
                p_y = col_sums[j]
                pmi = max(0, np.log(p_xy / p_x * p_y) + log_matrix_sum)
                pmi_matrix[i, j] = pmi
    # normalize row-wise
    pmi_matrix = pmi_matrix / np.linalg.norm(pmi_matrix, axis=1)[:, None]

    sparse_pmi_matrix = csr_matrix(pmi_matrix)
    rot_matrix = pmi_matrix.T
    fliped_pmi_matrix = csr_matrix(rot_matrix)

    return pmi_matrix, sparse_pmi_matrix, fliped_pmi_matrix

def sparse_matrix_to_dict(matrix):
    sparse_matrix = dict()
    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            if col > 0:
                if i not in sparse_matrix:
                    sparse_matrix[i] = dict()
                sparse_matrix[i][j] = col
    return sparse_matrix

def cosine(v1, v2):
    return np.dot(v1, v2) #/ (np.linalg.norm(v1) * np.linalg.norm(v2))

# calculate the similairy vector for word u
def cosine_with_sparse_matrix(matrix, fliped_matrix, u):
    dt = np.zeros(matrix.shape[0])
    # ATT(u) = fliped_matrix[u] (u'th row of fliped_matrix)
    # W(att) = matrix[att] (u'th row of matrix)
    for att in matrix[u].indices:
        for v in fliped_matrix[att].indices:
            dt[v] +=  matrix[u, att] * fliped_matrix[att, v]
    return dt

def cosine_similarity_matrix(pmi_matrix):
    cosine_matrix = np.zeros(pmi_matrix.shape)
    for i, row1 in enumerate(pmi_matrix):
        for j, row2 in enumerate(pmi_matrix):
            cosine_matrix[i, j] = cosine(row1, row2)
    return cosine_matrix

def calc_pmi(context_list, counts, words, word_to_idx, dep=False):
    # context_list = {k: v for k, v in context_list.items() if v >= 75}
    counts_matrix = vectorize_counts(counts, words, context_list, word_to_idx, dep=dep)
    print(f"shape of co-occurence matrix: {counts_matrix.shape}")
    pmi_matrix, sparse_pmi_matrix, fliped_pmi_matrix = pmi_from_matrix(counts_matrix)
    # cosine_matrix = cosine_similarity_matrix(pmi_matrix, words)
    # write_count_to_file(file_name, context_list, idx_to_words)
    return pmi_matrix, sparse_pmi_matrix, fliped_pmi_matrix

def k_most_similar(pmi_matrix, flipped_pmi_matrix, u, k=20, with_same=False):
    similarity_with_all = cosine_with_sparse_matrix(pmi_matrix, flipped_pmi_matrix, u)
    if not with_same:
        similarity_with_all[u] = -1
    x = np.argsort(similarity_with_all)[-k:] if with_same else np.argsort(similarity_with_all)[-k - 1 : -1]
    
    return list(reversed(x))
    
def calc_similar_to_words(words, pmi_matrix, flipped_pmi_matrix, word_to_idx):
    similarities_list = []
    for word in words:
        u = word_to_idx[word]
        similarities_list.append(k_most_similar(pmi_matrix, flipped_pmi_matrix, u))
    return similarities_list

def write_to_file_most_similars(similar_full_window, similar_2_window, similar_dep, words_to_test, idx_to_words):
    with open("top20.txt", "w") as f:
        for i, word in enumerate(words_to_test):
            f.write(f"{word}\n")
            for x, y, z in zip(similar_2_window[i], similar_full_window[i], similar_dep[i]):
                f.write(f"{idx_to_words[x]} {idx_to_words[y]} {idx_to_words[z]}\n")
            f.write(f"{'*' * 9}\n")

def filter_words(words, threshold = 100):
    new_words = {}
    for word, count in words.items():
        if count < threshold:
            break
        new_words[word] = count
    return new_words

def main():
    test_words = ['car','bus','hospital', 'hotel','gun', 'bomb','horse','fox','table','bowl', 'guitar', 'piano']

    file_name = sys.argv[1]
    df = read_table_from_file(file_name)

    # count words
    words = df["LEMMA"].value_counts()

    starts_indexes = df.index[df["ID"] == 1].tolist() + [len(df)]
    list_of_sentences = [df.iloc[starts_indexes[n] : starts_indexes[n + 1]] for n in range(len(starts_indexes) - 1)]

    words = filter_words(words, threshold=75)
    print(f"len of words after filter of 75: {len(words)}")
    # mapping word for memory efficiency
    idx_to_word = np.array([x[0] for x in words.items()])
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

    # apply first filter
    words = filter_words(words, threshold=100)
    print(f"len of words after filter of 100: {len(words)}")
    words = {word_to_idx[word]:count for word, count in words.items()}

    words_to_test = [word for word in test_words if word in word_to_idx and word_to_idx[word] in words]

    # write words to file
    write_count_to_file("counts_words.txt", words, idx_to_word)

    # count context1
    word_to_context_counts, context_count = context_counter(list_of_sentences, words, word_to_idx, task="full_window")
    pmi_matrix_full_window, sparse_matrix_full, fliped_sparse_matrix_full = calc_pmi(context_count, word_to_context_counts, words, word_to_idx)
    similariries_full_window = calc_similar_to_words(words_to_test, sparse_matrix_full, fliped_sparse_matrix_full, word_to_idx)

    # count context2
    word_to_context_counts2, context_count2 = context_counter(list_of_sentences, words, word_to_idx, task="2_window")
    pmi_matrix_two_window, sparse_matrix_two, fliped_sparse_matrix_two = calc_pmi(context_count2, word_to_context_counts2, words, word_to_idx)
    similariries_2_window = calc_similar_to_words(words_to_test, sparse_matrix_two, fliped_sparse_matrix_two, word_to_idx)

    # count context3
    word_to_context_counts3, context_count3 = context_counter(list_of_sentences, words, word_to_idx, task="dep_type")
    write_features_count_to_file("counts_context_dep.txt", context_count3, idx_to_word)
    pmi_matrix_dep, sparse_matrix_dep, fliped_sparse_matrix_dep = calc_pmi(context_count3, word_to_context_counts3, words, word_to_idx, dep=True)
    similariries_dep = calc_similar_to_words(words_to_test, sparse_matrix_dep, fliped_sparse_matrix_dep, word_to_idx)


    write_to_file_most_similars(similariries_full_window, similariries_2_window, similariries_dep, words_to_test, idx_to_word)


if __name__ == "__main__":
    main()