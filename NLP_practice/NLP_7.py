import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print(os.getcwd())
os.chdir('ML_study/NLP_practice')
device = torch.device("cuda")

def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

class word2vec:
    def __init__(self, embedding_dims, vocabulary):
        self.w1 = nn.Linear(embedding_dims, vocabulary_size)
        self.w2 = nn.Linear(vocabulary_size, embedding_dims)
    
    def forward(self,x):
        z1 = torch.matmul(self.w1, x)
        z2 = torch.matmul(self.w2, z1)



if __name__ == '__main__':
    corpus = [
                'he is a king',
                'she is a queen',
                'he is a man',
                'she is a woman',
                'warsaw is poland capital',
                'berlin is germany captital',
                'paris is france capital'
            ]
    print(corpus)
    tokenized_corpus = tokenize_corpus(corpus)
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    print(vocabulary)
    # indexing을 해서 어디에 어떤 단어인지 마킹 하는 게 중요함. 나중에 embedding을 해서 찾아야 해서
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx:w for (idx, w) in enumerate(vocabulary)}
    vocabulary_size = len(vocabulary)
    print(word2idx)
    print(idx2word)
    window_size = 2
    idx_pairs = []
    # he is a king -> center:'is' -> he is / is a / is king 나머지는 context
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size+1):
                context_word_pos = center_word_pos + w
                # make source not jump out sentence
                # 예외처리
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    idx_pairs = np.array(idx_pairs)
    print(idx_pairs)
    # P(context|center;\theta) center단어와 \theta 있을때 context의 확률을 구한다.
    # As we are interested in preditiing context given center word, we want to maximize P(context|center) for each centext, center pair
    # As sums up to 1
    learning_rate = 0.001
    epochs = 100
    embedding_dims = 5
    model = word2vec(embedding_dims, vocabulary_size)
    criterion = F.nll_loss(F.log_softmax)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        loss_val = 0
        for data, target in idx_pairs:
            optimizer.zero_grad()
            y_pre = model(data)
            loss = criterion(y_pre, traget)
            loss.backward()
            optimizer.step()
            print(epoch, loss.item())
