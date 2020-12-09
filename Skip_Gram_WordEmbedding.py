# Skip_Gram_WordEmbedding.py
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.manifold import TSNE

# 1. Define hyper-parameters
corpus_file = 'data/Harry Potter 1.txt'
size = 500 # Dimensionality of the word vectors.
window = 8 # Maximum distance between the current and predicted word within a sentence.
min_count = 5 # Ignores all words with total frequency lower than this.
sg = 1 # Training algorithm: 1 for skip-gram; otherwise CBOW.
epochs=10 # Number of iterations over the corpus, iter default 5.

# 2. Training model
model = Word2Vec(LineSentence(corpus_file), size=size, window=window,min_count=min_count,sg=sg,iter=epochs)

# 3. Calculate word embedding
def calculate_embedding(token):
    token_embedding=model.wv[token].tolist()
    print('\''+token + '\'','word_embedding: ', token_embedding)
    return token_embedding

# 4. Calculate words cos
def calculate_cos(token1,token2):
    calculate_embedding(token1)
    calculate_embedding(token2)
    cos=model.wv.similarity(token1,token2)
    print('\''+token1+'\' and \''+token2+'\'','cos similarity: ',cos)
    print("----------------------------------------------------------------")
    return cos

# 5. Visualize
def plot_with_labels(low_dim_embs,labels,filename):
    print('Visualizing.')
    plt.figure(figsize=(6,6))
    for i in range(len(labels)):
        x,y=low_dim_embs[i,:]
        label=labels[i]
        plt.scatter(x, y)
        plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
    plt.title("Skip_Gram_WordEmbedding", fontsize='10')
    plt.savefig(filename)
    plt.show()


print(model.wv.vectors.shape)
calculate_cos('Harry','Potter')
calculate_cos('Harry','Voldemort')
calculate_cos('man','woman')
calculate_cos('good','bad')
'''
'man' and 'woman' cos similarity:  0.93227917
'''

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
# perpexity 混乱度，表示t-SNE优化过程中考虑邻近点的多少，默认为30，建议取值在5到50之间
# n_components 嵌入空间的维度
# init初始化，默认为random，取值为pca为利用PCA进行初始化（常用）
# n_iter 迭代次数,默认为1000

labels1=['Harry','Hermione','man','woman'] #V(“Harry”) - V(“Hermione”)  ≈V(“man”) - V(“woman”)
embeddings1=[model.wv[label].tolist() for label in labels1]
low_dim_embs1 = tsne.fit_transform(embeddings1)
plot_with_labels(low_dim_embs1, labels1,'data/Skip_Gram_WordEmbedding1.png')


labels2=['Harry','Potter','Hermione','Granger']  #(Hermione Granger)(Harry Potter)
embeddings2=[model.wv[label].tolist() for label in labels2]
low_dim_embs2 = tsne.fit_transform(embeddings2)
plot_with_labels(low_dim_embs2, labels2,'data/Skip_Gram_WordEmbedding2.png')


def plot_epoch_model():
    epochs = [i for i in range(1, 21)]  # epoch: Number of iterations over the corpus.
    cos = []
    for epoch in epochs:
        model = Word2Vec(LineSentence(corpus_file), size=size, window=window, min_count=min_count, sg=sg, iter=epoch)
        cos.append(model.wv.similarity('Harry', 'Ron'))
        print("corpus epoch",epoch," finish")
    plt.figure(figsize=(8,8))
    plt.plot(epochs, cos, color='c', marker="o", markersize=2, linestyle='--', linewidth=1)
    for a, b in zip(epochs, cos):
        plt.text(a, b, (b), ha='left', va='bottom', fontsize=10)
    plt.title("Skip_Gram_WordEmbedding", fontsize=10)
    x_major_locator = MultipleLocator(2)  # 把x轴的刻度间隔设置为2，并存在变量里
    y_major_locator = MultipleLocator(0.1)  # 把y轴的刻度间隔设置为0.1，并存在变量里
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置
    plt.xlabel("corpus epoch", fontsize=10)
    plt.ylabel("\'Harry\' and \'Ron\' cos similarity", fontsize=10)

    plt.savefig('data/Skip_Gram_epoch_similarity.png')
    plt.show()

#plot_epoch_model()