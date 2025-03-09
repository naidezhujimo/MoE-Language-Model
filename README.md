# MoE-Language-Model
这是一个基于 稀疏混合专家（Sparse Mixture of Experts, MoE） 和 Transformer 架构的语言模型实现。该模型结合了多头注意力机制和 MoE 技术，能够高效地处理自然语言生成任务。代码使用 PyTorch 实现，并支持 GPU 加速。

项目特点
稀疏混合专家（MoE）：模型通过动态路由机制选择多个专家网络，每个专家专注于处理输入的不同部分，从而提高模型的表达能力和计算效率。

Transformer 架构：基于标准的 Transformer 模块，支持多头自注意力机制和前馈神经网络。

动态路由：使用带噪声的 Top-k 路由机制，动态选择最相关的专家网络。

负载均衡：通过辅助损失函数确保专家网络的负载均衡，避免某些专家被过度使用。

高效训练：支持学习率调度（余弦退火和 ReduceLROnPlateau），并提供详细的训练日志和损失曲线。

模型架构
嵌入层：

使用词嵌入和位置嵌入将输入序列映射到高维空间。

多头注意力机制：

每个注意力头独立计算注意力权重，捕捉输入序列中的上下文信息。

稀疏混合专家（MoE）：

每个 MoE 层包含多个专家网络，通过动态路由机制选择最相关的专家。

前馈神经网络：

每个专家网络是一个简单的两层全连接网络，用于处理输入特征。

生成文本：

支持 Top-p 采样和温度调节，生成多样化的文本。

使用方法
1. 安装依赖
确保已安装以下 Python 库：

bash
复制
pip install torch mlflow tiktoken tqdm matplotlib
2. 训练模型
运行以下命令开始训练：

bash
复制
python main.py --train
训练过程中，模型会定期保存检查点，并记录训练和验证损失。训练完成后，会生成损失曲线图 loss_curve.png。

3. 生成文本
训练完成后，可以使用以下命令生成文本：

bash
复制
python main.py --generate
生成的文本将保存到 generated_text.txt 文件中。

4. 超参数配置
可以在代码中调整以下超参数以优化模型性能：

n_embd：嵌入维度。

n_head：注意力头的数量。

n_layer：Transformer 层的数量。

num_experts：MoE 中专家的数量。

top_k：每个输入选择的专家数量。

batch_size：批量大小。

learning_rate：学习率。

dropout：Dropout 比例。

代码结构
模型架构：

Head：单头注意力机制。

MultiHeadAttention：多头注意力机制。

Expert：专家网络。

NoisyTopkRouter：带噪声的 Top-k 路由机制。

SparseMoE：稀疏混合专家模块。

Block：Transformer 块。

SparseMoELanguageModel：完整的语言模型。

训练与评估：

get_batch：从数据集中随机采样一批数据。

estimate_loss：计算训练和验证损失。

generate：生成文本。

工具函数：

decode：将 token ID 解码为文本。

kaiming_init_weights：使用 Kaiming 初始化方法初始化模型权重。

实验结果
训练过程中，模型会记录训练和验证损失，并生成损失曲线图。以下是一个示例损失曲线：

Loss Curve


参考
Attention is All You Need

Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

PyTorch Documentation
