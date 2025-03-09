import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import mlflow
import tiktoken
from tqdm import tqdm


#------------------------------------------------------------------------------
# 超参数设置
n_embd = 144 # 嵌入维度
n_head = 6 # 注意力头的数量
n_layer = 6 # Transformer 层的数量
head_size = 24 # 每个注意力头的大小
dropout = 0.2 # Dropout 比例
block_size = 48 # 模型处理的最大序列长度
num_experts = 6 # MoE 中专家的数量
top_k = 2 # 在 MoE 中，每个输入选择的专家数量
vocab_size = 50257 # 词汇表大小，表示模型可以处理的单词数量
batch_size = 96  # 每个批次的样本数量
max_iters = 6200  # 最大训练迭代次数
eval_interval = 100  # 每隔多少次迭代进行一次评估
eval_iters = 100  # 评估时使用的迭代次数
learning_rate = 3e-4  # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 训练设备

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding('gpt2') # 使用 GPT-2 的编解码器
tokens = enc.encode(text)
tokens = torch.tensor(tokens)
n = int(0.9*len(tokens))
train_data = tokens[:n]
val_data = tokens[n:]


# 注意力头模块
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # 三个线性变换层（分别为键、查询、值）
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 注册一个下三角矩阵 tril，用于实现因果注意力
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # 获取输入张量 x 的形状(批量大小、时间步长、特征维度)
        k = self.key(x) # 通过键变换层，得到键张量 k
        q = self.query(x) # 通过查询变换层，得到查询张量 q
        wei = q @ k.transpose(-2,-1) * C**-0.5 # 计算注意力权重矩阵 wei，即查询和键的点积，并乘以缩放因子 C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 将上三角部分的权重设置为负无穷
        wei = F.softmax(wei, dim=-1) # 对权重矩阵进行 Softmax 归一化
        wei = self.dropout(wei) # 应用 Dropout，防止过拟合
        v = self.value(x) # 通过值变换层，得到值张量 v
        out = wei @ v # 根据权重矩阵 wei 对值张量 v 进行加权求和
        return out
    
# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # 模块列表，每个头独立计算注意力
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 线性变换层，将多个头的输出拼接后投影到目标维度
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 将拼接后的输出通过投影层，将其维度转换为 (B, T, n_embd)并应用dropout
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# 专家模块
class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # 一个简单的前馈网络
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.layer_norm(x + self.net(x))
    
# 路由模块
class TopkRouter(nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k # 选择的专家数量
        self.linear = nn.Linear(n_embd, num_experts) # 线性变换层，将输入映射到专家数量的维度

    def forward(self, mh_output):
        logits = self.linear(mh_output) # 得到每个输入对每个专家的 logits
        top_k_logits, indices = logits.topk(self.top_k, dim=-1) # 选择 top-k 个专家的 logits 和对应的索引
        zeros = torch.full_like(logits, float('-inf')) # 创建一个与 logits 形状相同的张量
        sparse_logits = zeros.scatter(-1, indices, top_k_logits) # 将 top-k 的 logits 填充到负无穷张量中，其余位置保持负无穷
        router_output = F.softmax(sparse_logits, dim=-1) # 对稀疏 logits 应用 Softmax，得到每个输入对每个专家的权重
        return router_output, indices # 返回路由权重和选择的专家索引

# 定义噪声路由模块
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embd, num_experts) # 用于计算路由
        self.noise_linear = nn.Linear(n_embd, num_experts) # 用于生成噪声
        self.temperature = 1.0  # 可学习参数
        # 使用更小的初始化方差
        init.normal_(self.topkroute_linear.weight, mean=0.0, std=0.02)
        init.normal_(self.noise_linear.weight, mean=0.0, std=0.02)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits) # 从正态分布中随机生成噪声并与正数的缩放因子相乘
        noise_logits = logits + self.temperature * noise # 将噪声添加到路由 logits 中

        top_k_logits, indices = noise_logits.topk(self.top_k, dim=-1) # 对带噪声的 logits 选择 top-k 个专家
        zeros = torch.full_like(noise_logits, float('-inf')) # 创建一个与 logits 形状相同的张量
        sparse_logits = zeros.scatter(-1, indices, top_k_logits) # 将 top-k 的 logits 填充到负无穷张量中，其余位置保持负无穷
        router_output = F.softmax(sparse_logits, dim=-1) # 对 sparse_logits 应用 Softmax
        return router_output, indices # 返回路由权重和选择的专家索引
    
# 稀疏混合专家模块
class SparseMoE(nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embd, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.top_k = top_k
        self.num_experts = num_experts
        # 专家负载均衡相关
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.aux_loss_weight = 0.02 # 辅助损失权重

    def forward(self, x):
        B, T, C = x.shape
        gating, indices = self.router(x)

        # 动态更新专家选择统计
        self.expert_counts += torch.histc(
            indices.flatten().float(), 
            bins=self.num_experts,
            min=0,
            max=self.num_experts-1
        )

        # 当前batch的专家选择统计
        indices_flat = indices.view(-1)
        current_counts = torch.bincount(indices_flat, minlength=self.num_experts).float().to(device)
        
        # 专家利用率计算
        expert_usage = (current_counts > 0).float().mean()  # 计算至少被选择一次的专家的比例
        self.expert_usage = expert_usage.detach()

        # 计算负载均衡损失(平滑处理)
        expert_probs = (current_counts + 1e-3) / (current_counts.sum() + 1e-3 * self.num_experts)
        uniform_dist = torch.ones_like(expert_probs) / self.num_experts
        
        # KL散度计算
        self.aux_loss = F.kl_div(
            input=torch.log(expert_probs + 1e-10),  # 输入为模型分布的对数概率
            target=uniform_dist,                    # 目标为均匀分布
            reduction='batchmean'
        )
        
        # 展平输入
        flat_x = x.view(-1, C)
        final_output = torch.zeros_like(flat_x)
        
        # 遍历每个专家
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1).view(-1)
            if expert_mask.any():
                expert_input = flat_x[expert_mask]
                expert_output = expert(expert_input)
                gating_scores = gating.view(-1, self.num_experts)[expert_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
        
        # 恢复形状
        return final_output.view(B, T, C)
            
# Transformer块
class Block(nn.Module):
    def __init__(self, n_embd, n_head, num_experts, top_k):
        super().__init__()
        head_size = n_embd // n_head # 计算每个注意力头的大小
        self.sa = MultiHeadAttention(n_head, head_size) # 多头注意力模块
        self.smoe = SparseMoE(n_embd, num_experts, top_k) # 稀疏混合专家模块
        # 定义两个层归一化（LayerNorm）模块，用于稳定训练
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x

# 语言模型
class SparseMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 词嵌入表，将单词索引映射到嵌入向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置嵌入表，为每个位置添加位置信息
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, num_experts=num_experts,top_k=top_k) for _ in range(n_layer)]) # 模块序列
        self.ln_f = nn.LayerNorm(n_embd) # 最终的层归一化模块
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性变换层，将嵌入向量映射到词汇表大小的维度，用于生成下一个单词的概率分布

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # 将输入索引通过词嵌入表，得到词嵌入
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # 为每个位置生成位置嵌入
        x = tok_emb + pos_emb # 将词嵌入和位置嵌入相加，得到输入张量
        aux_loss_total = 0.0
        for block in self.blocks:
            x = block(x)
            aux_loss_total += block.smoe.aux_loss * block.smoe.aux_loss_weight
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 将 logits 展平为 (B * T, C)
            targets = targets.view(B*T) # 将目标标签展平为 (B * T)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1) # 计算交叉熵损失

        return logits, loss, aux_loss_total
        
    def generate(self, idx, max_new_tokens, top_p=0.9, temperature=1.0):
        for _ in tqdm(range(max_new_tokens), desc='Generate Text', unit='iter'):
            # 裁剪输入序列，确保不超过 block_size
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
            
            # 获取模型输出
            logits, _, _ = self(idx_cond)
            
            # 取最后一个时间步的 logits
            logits = logits[:, -1, :] / temperature  # 应用温度参数
            
            # 计算概率分布
            probs = F.softmax(logits, dim=-1)
            
            # Top-p 采样
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 将不需要的 token 的概率设为 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            
            # 重新归一化概率分布
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 从剩余 token 中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 将生成的 token 添加到输入序列中
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

model = SparseMoELanguageModel()
model.apply(_init_weights) 

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') #  打印模型的参数数量

max_lr = learning_rate # 最大学习率
min_lr = max_lr * 0.1 # 最小学习率
warmup_steps = 1000 # 学习率预热步数

from torch.optim.lr_scheduler import CosineAnnealingLR
# 使用 AdamW 优化器初始化模型参数
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), eps=1e-8)
# 定义余弦退火调度器
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=min_lr)


def get_batch(data, block_size, batch_size, device):
    # 检查数据长度是否足够
    assert len(data) > block_size, f"数据长度不足: len(data)={len(data)}, block_size={block_size}"
    
    # 随机生成 batch_size 个起始索引
    idxs = torch.randint(0, len(data) - block_size, (batch_size,))
    
    # 提取输入序列和目标序列
    xb = torch.stack([data[i:i+block_size] for i in idxs])
    yb = torch.stack([data[i+1:i+block_size+1] for i in idxs])
    
    # 移动到设备
    xb = xb.to(device)
    yb = yb.to(device)
    
    # 添加token dropping (10%概率)
    drop_mask = torch.rand_like(xb.float()) < 0.1
    xb[drop_mask] = enc.encode(
        "<|endoftext|>", 
        allowed_special={"<|endoftext|>"}  # 允许使用特殊token
    )[0]  # 用特殊标记替换
    
    return xb, yb

# 损失估计函数
def estimate_loss(model, eval_iters, device, train_data, val_data, block_size, batch_size, get_batch):
    model.eval() # 将模型切换到评估模式
    train_losses = [] # 训练集损失
    val_losses = [] # 验证集损失
    with torch.no_grad():  # 确保在评估时不计算梯度
        for _ in range(eval_iters):
            xb, yb = get_batch(train_data, block_size, batch_size, device)
            logits, loss, aux_loss = model(xb, yb) # 将输入传递给模型，计算 logits 和损失
            train_losses.append(loss.item())
        for _ in range(eval_iters):
            xb, yb = get_batch(val_data, block_size, batch_size, device)
            logits, loss, aux_loss = model(xb, yb)
            val_losses.append(loss.item())
    model.train() # 将模型切换回训练模式
    return {"train": torch.tensor(train_losses).mean().item(), "val": torch.tensor(val_losses).mean().item()}

# 解码函数
def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()  # 如果 ids 是张量，转换为列表
    return enc.decode(ids)  # 使用 tiktoken 解码


checkpoint_path = "model_checkpoint.pth" # 定义模型检查点的保存路径
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="Sparse MoE Language Model")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--generate", action="store_true", help="Generate text using the trained model")
args = parser.parse_args()

import matplotlib.pyplot as plt
import seaborn as sns

# 添加损失值记录
train_losses = []  # 记录训练集损失
val_losses = []    # 记录验证集损失

# 训练逻辑
if args.train:
    print("Starting training...")
    with mlflow.start_run():
        params = {
            "batch_size": batch_size,
            "block_size": block_size,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "learning_rate": learning_rate,
            "device": device,
            "eval_iters": eval_iters,
            "dropout": dropout,
            "num_experts": num_experts,
            "top_k": top_k
        }
        mlflow.log_params(params)  # 将超参数记录到 mlflow
        for iter in tqdm(range(max_iters), desc="Training", unit="iter"):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                losses = estimate_loss(model, eval_iters, device, train_data, val_data, block_size, batch_size, get_batch)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # 打印专家利用率
                avg_usage = sum(block.smoe.expert_usage for block in model.blocks) / n_layer
                print(f"Expert Usage: {avg_usage:.2%}")
                # 打印梯度范数
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient Norm: {total_norm:.2f}")

                metrics = {"train_loss": losses['train'], "val_loss": losses['val']}

                # 记录损失值
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

                # 打印当前学习率
                print(f"Iter {iter}: Current LR = {optimizer.param_groups[0]['lr']}")

                mlflow.log_metric("train_loss", losses['train'], step=iter)
                mlflow.log_metric("val_loss", losses['val'], step=iter)

            if iter % (eval_interval*10) == 0:  # 每10次评估保存一次热力图
                expert_counts = torch.stack(
                    [block.smoe.expert_counts.cpu() for block in model.blocks]
                ).float().numpy()
                
                plt.figure(figsize=(10,6))
                sns.heatmap(expert_counts, annot=True)
                plt.savefig(f"expert_heatmap_{iter}.png")
                plt.close()  # 关闭图像释放内存

            xb, yb = get_batch(train_data, block_size, batch_size, device)  # 从训练集中采样一批数据
            logits, loss, aux_loss = model(xb, yb)
            total_loss = loss + aux_loss # 集成辅助损失
            optimizer.zero_grad()  # 清零梯度
            total_loss.backward()  # 反向传播，计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()  # 更新模型参数
            # 更新余弦退火调度器
            scheduler.step()

            if iter % eval_interval == 0 or iter == max_iters - 1:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        # 训练完成后绘制损失曲线
        iterations = list(range(0, max_iters+1, eval_interval))
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_losses, label="Train Loss", color="blue")
        plt.plot(iterations, val_losses, label="Validation Loss", color="red")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve.png")  # 保存图像
        plt.show()  # 显示图像

# 推理逻辑
if args.generate:
    print("Loading model weights and generating text...")
    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式

    # 生成文本
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=2000, top_p=0.9, temperature=0.8)[0]
    generated_text = decode(generated_tokens)
    print(generated_text)
    output_file = "generated_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")
