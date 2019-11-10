from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import pandas


def read_vec(x):
    result = {}
    for l in x[2:]:
        w2v = l.split()
        w = w2v[0]
        v = list(map(float, tuple(w2v[1:])))
        result.update({w: v})
    return result


ant_file = open(r'D:\Python shit\Word-Similarity\antonym-synonym set\Antonym_vietnamese.txt', 'r', encoding='utf8')
ant_lines = ant_file.readlines()

pre_trained_file = open(r'D:\Python shit\Word-Similarity\word2vec\W2V_150.txt', 'r', encoding='utf8')
pre_trained_lines = pre_trained_file.readlines()

w2v_dict = read_vec(pre_trained_lines)
pre_trained_file.close()

x_train = []
y_train = []
feature1 = []
for line in ant_lines:
    words = line.split()
    if len(words) == 2:
        word1 = words[0]
        word2 = words[1]
        vec1 = w2v_dict.get(word1)
        vec2 = w2v_dict.get(word2)
        if vec1 is not None and vec2 is not None:
            x_train.append(vec1 + vec2)
            y_train.append(-1)

ant_file.close()

sym_file = open(r'D:\Python shit\Word-Similarity\antonym-synonym set\Synonym_vietnamese.txt', 'r', encoding='utf8')
sym_lines = sym_file.readlines()

for line in sym_lines:
    words = line.split()
    if len(words) == 2:
        word1 = words[0]
        word2 = words[1]
        vec1 = w2v_dict.get(word1)
        vec2 = w2v_dict.get(word2)
        if vec1 is not None and vec2 is not None:
            x_train.append()
            y_train.append(1)

sym_file.close()

x_train = pandas.DataFrame(x_train)
model = MLPClassifier()
model.fit(x_train, y_train)
print(len(y_train))
x_tests = []
y_grounds = []


def read_test_data(x):
    for l in x[1:]:
        stat = l.split()
        w1 = stat[0]
        w2 = stat[1]
        v1 = w2v_dict.get(w1)
        v2 = w2v_dict.get(w2)
        if v1 is not None and v2 is not None:
            x_tests.append(v1 + v2)
            y_grounds.append(-1 if stat[2] == 'ANT' else 1)


vicon400_noun_file = open(r'D:\Python shit\Word-Similarity\datasets\ViCon-400\400_noun_pairs.txt', 'r', encoding='utf8')
vicon400_noun_lines = vicon400_noun_file.readlines()
read_test_data(vicon400_noun_lines)
vicon400_noun_file.close()

vicon400_verb_file = open(r'D:\Python shit\Word-Similarity\datasets\ViCon-400\400_verb_pairs.txt', 'r', encoding='utf8')
vicon400_verb_lines = vicon400_verb_file.readlines()
read_test_data(vicon400_verb_lines)
vicon400_verb_file.close()

vicon400_adj_file = open(r'D:\Python shit\Word-Similarity\datasets\ViCon-400\600_adj_pairs.txt', 'r', encoding='utf8')
vicon400_adj_lines = vicon400_adj_file.readlines()
read_test_data(vicon400_adj_lines)
vicon400_adj_file.close()

x_tests = pandas.DataFrame(x_tests)
y_tests = model.predict(x_tests)
print(x_tests.shape)

print(metrics.confusion_matrix(y_grounds, y_tests))
print(metrics.precision_score(y_grounds, y_tests))
print(metrics.recall_score(y_grounds, y_tests))
print(metrics.f1_score(y_grounds, y_tests))
