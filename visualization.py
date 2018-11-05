from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

vocab = list(austen_w2v.wv.vocab)
X = austen_w2v[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

tsne_df = pd.DataFrame(X_tsne, index = vocab, columns = ['x','y'])

fig = sns.scatterplot(tsne_df['x'],tsne_df['y'])
for word, pos in tsne_df.iterrows():
    fig.annotate(word, pos)

plt.show()

plt.ylim(40,80)
plt.xlim(-25,50)

def tsne_closest(model, word):
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    #similar words
    close_words = model.wv.most_similar(word, topn=30)

    #vector for close words
    arr = np.append(arr, np.array([model[word]]),axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    #tsne coords
    tsne = TSNE(n_components=2, random_state=748)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    #display
    sns.scatterplot(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy = (x,y), xytext = (0,0), textcoords='offset points')
    plt.plot(x_coords[0],y_coords[0],'ro')
    plt.show()

tsne_closest(austen_w2v,"pride")