"""
@File  :t-sne.py
@Author:Miles
@Date  :2020/11/1413:42
@Desc  :
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


# load data
def get_data():
    """
    :return: dataset, labels, number of samples, number of features
    """
    digits = datasets.load_digits(n_class=10)
    data = digits.data  # image features
    label = digits.target  # Image tags
    n_samples, n_features = data.shape  # shape of dataset
    return data, label, n_samples, n_features


def get_feature(np_dir='./visualization/morph/train_baseline_feature.npz'):
    morph_pml = '../visualization/morph/train_pml_feature.npz'
    morph_base = '../visualization/morph/train_baseline_feature.npz'
    np_dir = morph_pml
    a = np.load(np_dir)
    data = a['feature']
    label = a['label']
    # head_index = [20, 24, 28, 32, 36, 40,]
    index = [20, 24, 28, 32, 36, 40, 54, 57, 60, 63, 66, 69]

    for i in index:
        index_ = np.where(label == i)

        # for test
        # index_max = int(int(index_[0].shape[0]) / 3)
        # index_ = index_[0][0:index_max]

        label_ = label[index_]
        data_ = np.squeeze(data[index_, :])
        if i == 20:
            label_select = label_
            data_select = data_
        else:
            label_select = np.concatenate([label_select, label_])
            data_select = np.concatenate([data_select, data_], axis=0)
    n_samples, feature_dim = data_select.shape
    return data_select, label_select, n_samples, feature_dim


# Preprocess and draw the sample
def plot_embedding(data, label, title):
    """
    :param data:dataset
    :param label:sample label
    :param title:image caption
    :return:image
    """
    plt.rcParams['savefig.dpi'] = 300  # picture pixel
    plt.rcParams['figure.dpi'] = 300  # Resolution
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # Normalize the data
    fig = plt.figure()  # Create a graph instance
    ax = plt.subplot(111)  # Create a subgraph
    # iterate over all samples
    for i in range(data.shape[0]):
        # Draw labels for each data point in the graph
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
        #          fontdict={'weight': 'bold', 'size': 7})
        if label[i] == 20:
            col = plt.cm.Set1(0)
        if label[i] == 24:
            col = plt.cm.Set1(1)
        if label[i] == 28:
            col = plt.cm.Set1(2)
        if label[i] == 32:
            col = plt.cm.Set1(3)
        if label[i] == 36:
            col = plt.cm.Set1(4)
        if label[i] == 40:
            col = plt.cm.Set1(5)

        if label[i] == 54:
            col = plt.cm.Set1(6)
        if label[i] == 57:
            col = plt.cm.Set1(7)
        if label[i] == 60:
            col = plt.cm.Set1(8)
        if label[i] == 63:
            col = plt.cm.Set2(0)
        if label[i] == 66:
            col = plt.cm.Set2(1)
        if label[i] == 69:
            col = plt.cm.Set2(2)
        plt.scatter(data[i, 0], data[i, 1], s=2, color=col)
    plt.xticks()  # scale of the specified coordinates
    plt.yticks()
    # plt.title(title, fontsize=14)
    # return value
    return fig


# Main function, which performs t-SNE dimensionality reduction
def main():
    data, label, n_samples, n_features = get_feature()  # Call the function to get the dataset information
    print('Starting compute t-SNE Embedding...')
    # ts = TSNE(n_components=2, init='pca', random_state=0, perplexity=80, verbose=True, n_iter=500)
    ts = TSNE(n_components=2, init='pca', random_state=0, perplexity=20, verbose=True, n_iter=1100)
    # t-SNE Dimensionality reduction
    reslut = ts.fit_transform(data)
    # Call the function to draw the image
    fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
    # display image
    plt.show()


# main function
if __name__ == '__main__':
    main()
