from scipy import spatial
import os


def read_vec(x):
    result = {}
    for l in x[2:]:
        w2v = l.split()
        w = w2v[0]
        v = list(map(float, w2v[1:]))
        result.update({w: v})
    return result


def k_nn(x, k):
    relations = {}
    x_vec = w2v_dict.get(x)
    for w in w2v_dict.keys():
        v = w2v_dict.get(w)
        relations.update({spatial.distance.cosine(x_vec, v): w})
    print('Word:', x, '\nSimilarity, word: ',
          [[1 - key, relations.get(key)] for key in sorted(relations.keys())[1:k]])


fileDir = os.path.dirname(os.path.abspath(__file__))
print(fileDir)
parentDir = os.path.dirname(fileDir)
print(parentDir)
pre_trained = open(parentDir + r'\word2vec\W2V_150.txt', 'r', encoding='utf8')
pre_trained_lines = pre_trained.readlines()
w2v_dict = read_vec(pre_trained_lines)

pre_trained.close()

k_nn('và', 10)

