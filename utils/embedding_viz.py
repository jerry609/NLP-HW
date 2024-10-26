# embedding_utils.py

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

def load_all_embeddings(file_path, expected_dim=None):
    """
    加载词向量文件中的所有词汇和向量。

    Args:
        file_path: 词向量文件路径。
        expected_dim: 预期的向量维度。

    Returns:
        word_list: 词汇列表。
        embeddings: 词向量矩阵。
    """
    word_list = []
    embeddings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            values = line.strip().split()
            if len(values) < 2:
                print(f"警告：文件第 {line_num} 行格式错误，已跳过。")
                continue
            word = values[0]
            vector_values = values[1:]
            if expected_dim is None:
                expected_dim = len(vector_values)
            if len(vector_values) != expected_dim:
                print(f"警告：文件第 {line_num} 行的向量维度不匹配，已跳过。")
                print(f"问题行内容：{line.strip()}")
                continue
            try:
                vector = np.array(vector_values, dtype=float)
                word_list.append(word)
                embeddings.append(vector)
            except ValueError:
                print(f"警告：文件第 {line_num} 行包含无法转换的值，已跳过。")
                print(f"问题行内容：{line.strip()}")
    embeddings = np.array(embeddings)
    return word_list, embeddings

def get_embeddings_for_selected_words(word_list, embeddings, selected_words):
    """
    获取选定词汇的向量。

    Args:
        word_list: 所有词汇的列表。
        embeddings: 所有词汇的向量矩阵。
        selected_words: 选定的词汇列表。

    Returns:
        words_found: 实际找到的词汇列表（可能少于 selected_words）。
        selected_embeddings: 选定词汇的向量矩阵。
    """
    word_to_index = {word: idx for idx, word in enumerate(word_list)}
    selected_embeddings = []
    words_found = []
    for word in selected_words:
        idx = word_to_index.get(word)
        if idx is not None:
            selected_embeddings.append(embeddings[idx])
            words_found.append(word)
        else:
            print(f"警告：词汇 '{word}' 未在词汇表中找到，已跳过。")
    selected_embeddings = np.array(selected_embeddings)
    return words_found, selected_embeddings

def select_random_words(word_list, n=100):
    """
    随机选择 n 个词汇。

    Args:
        word_list: 所有词汇的列表。
        n: 要选择的词汇数量。

    Returns:
        selected_words: 选择的词汇列表。
    """
    if n > len(word_list):
        n = len(word_list)
    selected_words = random.sample(word_list, n)
    return selected_words

def select_top_n_words(word_list, n=100):
    """
    选择前 n 个词汇。

    Args:
        word_list: 所有词汇的列表。
        n: 要选择的词汇数量。

    Returns:
        selected_words: 选择的词汇列表。
    """
    selected_words = word_list[:n]
    return selected_words

def reduce_dimensions_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000):
    """
    使用 t-SNE 进行降维。

    Args:
        embeddings: 高维词向量矩阵。
        n_components: 降维后的维度。
        perplexity: t-SNE 的 perplexity 参数。
        n_iter: t-SNE 的迭代次数。

    Returns:
        reduced_embeddings: 降维后的词向量矩阵。
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings(words, embeddings, title='Word Embeddings Visualization', save_path=None):
    """
    绘制词向量的散点图。

    Args:
        words: 词汇列表。
        embeddings: 词向量矩阵（二维）。
        title: 图形标题。
        save_path: 图片保存路径。如果为 None，则不保存。

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    for i, word in enumerate(words):
        x, y = embeddings[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=9)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存为 {save_path}")
    plt.show()
