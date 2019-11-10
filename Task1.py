from scipy import spatial
from scipy import stats
import pandas


def read_vec(x):
    result = {}
    for l in x[2:]:
        w2v = l.split()
        w = w2v[0]
        v = list(map(float, w2v[1:]))
        w2v.update({w: v})
    return result


pre_trained_file = open(r'D:\Python shit\Word-Similarity\word2vec\W2V_150.txt', 'r', encoding='utf8')
pre_trained_lines = pre_trained_file.readlines()
w2v_dict = read_vec(pre_trained_lines)

pre_trained_file.close()

visim400 = open(r'D:\Python shit\Word-Similarity\datasets\ViSim-400\Visim-400.txt', 'r', encoding='utf8')
visim400_lines = visim400.readlines()
data = []
i = 0
for line in visim400_lines[1:]:
    stat = line.split()
    word1 = stat[0]
    word2 = stat[1]
    vec1 = w2v_dict.get(word1)
    vec2 = w2v_dict.get(word2)
    if vec1 is not None and vec2 is not None:
        cosine = spatial.distance.cosine(vec1, vec2)
        data.append([word1, word2, 3*cosine, stat[3]])

dataset = pandas.DataFrame(data, columns=['Word 1', 'Word 2', 'W2V Cosine', 'ViSim Cosine'])
dataset['ViSim Cosine'] = dataset['ViSim Cosine'].astype(float)
print('SpeanmanR:', stats.spearmanr(dataset['W2V Cosine'], dataset['ViSim Cosine']))
print('PearsonR:', stats.pearsonr(dataset['W2V Cosine'], dataset['ViSim Cosine']))
#result.to_csv(r'D:\Python shit\Word-Similarity\Tasks\Table1.csv', encoding='utf8')
visim400.close()
