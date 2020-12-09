# Bert_WordEmbedding.py
from transformers import BertTokenizer, BertModel
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. Load model.
def load_model():
    model_name = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model,tokenizer

model,tokenizer=load_model()

# 2. Data preprocessing.
def data_pre(token):
    token_input = tokenizer(token, return_tensors='pt')
    token_decode = tokenizer.decode(token_input['input_ids'][0])
    return token_input,token_decode

# 3. Calculate word embedding
def calculate_embedding(token):
    token_input, token_decode = data_pre(token)
    model.eval()
    token_embedding, _ = model(**token_input)
    token_embedding = token_embedding.squeeze(0)
    token_embedding = token_embedding[1, :]
    token_embedding = [i for i in token_embedding.detach().numpy().tolist()]
    print('\'' + token + '\'', 'word_embedding: ', token_embedding)
    return token_embedding

# 4. Calculate words cos
def calculate_cos(token1,token2):
    token_embedding1 = calculate_embedding(token1)
    token_embedding2 = calculate_embedding(token2)
    print('len(token_embedding1): ',len(token_embedding1))
    print('len(token_embedding2): ',len(token_embedding2))
    inner_product= 0
    module_length_1=0
    module_length_2=0
    for i in range(len(token_embedding1)):
        inner_product += token_embedding1[i]*token_embedding2[i]
        module_length_1+=token_embedding1[i]*token_embedding1[i]
        module_length_2+=token_embedding2[i]*token_embedding2[i]
    module_length_1=math.sqrt(module_length_1)
    module_length_2 = math.sqrt(module_length_2)
    cos = inner_product/(module_length_1*module_length_2)
    print('\''+token1+'\' and \''+token2+'\'','cos similarity: ',cos)
    print("----------------------------------------------------------------")
    return cos

# 5. Visualize
def plot_with_labels(low_dim_embs, labels,filename):
    print('Visualizing.')
    plt.figure(figsize=(6,6))
    for i in range(len(labels)):
        x,y=low_dim_embs[i,:]
        label=labels[i]
        plt.scatter(x, y)
        plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
    plt.title("Bert_WordEmbedding", fontsize='10')
    plt.savefig(filename)
    plt.show()


#token_embedding=calculate_embedding('king')
calculate_cos('king','queen')
calculate_cos('man','woman')
calculate_cos('king','man')
calculate_cos('queen','woman')

'''
'man' and 'woman' cos similarity:  0.8619070037354551
'''

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
# perpexity 混乱度，表示t-SNE优化过程中考虑邻近点的多少，默认为30，建议取值在5到50之间
# n_components 嵌入空间的维度
# init初始化，默认为random，取值为pca为利用PCA进行初始化（常用）
# n_iter 迭代次数,默认为1000

labels=['king','queen','man','woman']  #V(“king”) - V(“queen”)  ≈V(“man”) - V(“woman”)
embeddings=[calculate_embedding(label) for label in labels]
low_dim_embs = tsne.fit_transform(embeddings)
plot_with_labels(low_dim_embs, labels,'data/Bert_WordEmbedding.png')