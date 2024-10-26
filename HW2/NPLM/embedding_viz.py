import numpy as np
from sklearn.decomposition import PCA
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

def plot_embeddings(words, embeddings, title='Word Embeddings Visualization', save_path=None):
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

def reduce_dimensions_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

# ----------------------------
# 加载中文词向量并可视化
# ----------------------------
# 1. 加载词向量文件并提取词汇列表
chinese_file_path = 'chinese_embeddings.txt'  # 替换为您的中文词向量文件路径
embedding_dim = 100  # 根据实际的向量维度设置

word_list_ch, embeddings_ch = load_all_embeddings(chinese_file_path, expected_dim=embedding_dim)

# 检查是否成功加载
print(f"成功加载了 {len(word_list_ch)} 个中文词汇的向量。")

# 如果加载成功，继续执行后续步骤
if len(word_list_ch) > 0:
    # 2. 根据某种策略选择词汇
    # 策略 1：选择前 100 个高频词
    selected_words_ch = select_top_n_words(word_list_ch, n=100)

    # 3. 提取选定词汇的向量
    words_found_ch, selected_embeddings_ch = get_embeddings_for_selected_words(word_list_ch, embeddings_ch, selected_words_ch)

    # 4. 降维处理（使用 t-SNE 或 PCA）
    reduced_embeddings_ch = reduce_dimensions_tsne(selected_embeddings_ch)

    # 5. 绘制图形并保存图片
    plot_embeddings(words_found_ch, reduced_embeddings_ch, title='Chinese Word Embeddings Visualization', save_path='chinese_embeddings.png')
else:
    print("未能成功加载任何中文词汇的向量，请检查词向量文件。")

# ----------------------------
# 加载英文词向量并可视化
# ----------------------------
# 1. 加载词向量文件并提取词汇列表
english_file_path = 'english_embeddings.txt'  # 替换为您的英文词向量文件路径

word_list_en, embeddings_en = load_all_embeddings(english_file_path, expected_dim=embedding_dim)

# 检查是否成功加载
print(f"成功加载了 {len(word_list_en)} 个英文词汇的向量。")

# 如果加载成功，继续执行后续步骤
if len(word_list_en) > 0:
    # 2. 根据某种策略选择词汇
    # 策略 1：选择前 100 个高频词
    selected_words_en = select_top_n_words(word_list_en, n=100)

    # 3. 提取选定词汇的向量
    words_found_en, selected_embeddings_en = get_embeddings_for_selected_words(word_list_en, embeddings_en, selected_words_en)

    # 4. 降维处理（使用 t-SNE 或 PCA）
    reduced_embeddings_en = reduce_dimensions_tsne(selected_embeddings_en)

    # 5. 绘制图形并保存图片
    plot_embeddings(words_found_en, reduced_embeddings_en, title='English Word Embeddings Visualization', save_path='english_embeddings.png')
else:
    print("未能成功加载任何英文词汇的向量，请检查词向量文件。")
