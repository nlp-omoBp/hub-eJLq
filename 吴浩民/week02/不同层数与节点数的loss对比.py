# 数据处理专家: 专门用于表格数据的读取、清洗、分析和处理
import pandas as pd
# 深度学习框架核心: 提供张量计算和自动微分的基础功能
import torch
# 神经网络工具箱: 提供各种神经网络层、损失函数和模型容器
import torch.nn as nn
# 优化算法库: 实现各种梯度下降优化算法
import torch.optim as optim
# 数据加载系统: 提供数据集管理和批量数据加载功能
from torch.utils.data import Dataset, DataLoader
# 绘图库: 用于可视化不同模型的 Loss 变化对比
import matplotlib.pyplot as plt

# --- 1. 数据预处理 (保持不变) ---
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(sorted(set(labels)))}  # 使用 sorted 保证映射稳定
numerical_labels = [label_to_index[label] for label in labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_label = {i: label for label, i in label_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40


# --- 2. 构建 Dataset (保持不变) ---
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = labels
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0: bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)


# --- 3. 动态神经网络模型: 支持自定义层数和节点数 ---
class DynamicClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        # hidden_layers 是一个列表，例如 [256, 128] 代表两层隐藏层
        super(DynamicClassifier, self).__init__()
        layers = []
        current_dim = input_dim

        # 动态构建隐藏层序列
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim

        # 将列表转化为顺序容器
        self.feature_extractor = nn.Sequential(*layers)
        # 最终输出层
        self.classifier = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        out = self.feature_extractor(x)
        return self.classifier(out)


# --- 4. 实验对比逻辑 ---

# 定义三种不同的“大脑配置”进行 PK
# 1. Baseline: 只有一层 128 节点
# 2. Wide: 只有一层，但节点非常多 (1024)
# 3. Deep: 层数多，逐步压缩特征 (128 -> 64 -> 32)
configs = {
    "Baseline (128)": [128],
    "Wide (1024)": [1024],
    "Deep (128-64-32)": [128, 64, 32]
}

output_dim = len(label_to_index)
num_epochs = 15
all_results = {}  # 用于存放每个模型的 Loss 历史

for name, hidden_config in configs.items():
    print(f"\n--- 正在实验模型: {name} ---")
    model = DynamicClassifier(vocab_size, hidden_config, output_dim)
    criterion = nn.CrossEntropyLoss()
    # 使用 Adam 优化器，能更客观地在不同结构间对比收敛速度
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

    all_results[name] = history

# --- 5. 可视化对比 ---

plt.figure(figsize=(10, 6))
for name, losses in all_results.items():
    plt.plot(range(1, num_epochs + 1), losses, label=name, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.title('Comparison of Different Model Architectures')
plt.legend()
plt.grid(True)
plt.show()


# --- 6. 最后的预测演示 (使用最后训练的模型) ---
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 模拟单个样本的处理逻辑
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0: bow_vector[index] += 1

    model.eval()
    with torch.no_grad():
        output = model(bow_vector.unsqueeze(0))
    _, pred = torch.max(output, 1)
    return index_to_label[pred.item()]


test_text = "今天天气怎么样"
res = classify_text(test_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"\n最终测试: '{test_text}' -> 预测类别: {res}")
