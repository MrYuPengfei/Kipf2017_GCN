"""PyTorch 版层定义（本项目的注释基准文件，逐行注释）。

对照 TensorFlow 原版 ``gcn/layers.py`` 移植，保持相同的计算图语义：
  - Dense            ： dropout -> 线性变换 -> (+bias) -> 激活
  - GraphConvolution ： dropout -> 按 support 个数分别线性变换
                                -> support 前向 -> 求和 -> (+bias) -> 激活

两条链路的形状约定
------------------
设输入 ``X`` 形状为 ``(N, F_in)``，support 矩阵 ``S_i`` 形状为 ``(N, N)``，
权重 ``W_i`` 形状为 ``(F_in, F_out)``：

    pre_sup = X · W_i                -> (N, F_out)
    support = S_i · pre_sup        -> (N, F_out)
    output  = Σ_i support_i         -> (N, F_out)

GCN 取 K=1（单个 support）；ChebNet 取 K=max_degree（K+1 个 support），
两者用的是**同一个** GraphConvolution 类，差别仅在 support 个数与个数对应的权重。

关键差异：TF 用 ``tf.sparse_placeholder`` 接收稀疏输入，这里改用
``torch.sparse_coo_tensor``；稀疏 dropout 通过对稀疏张量的 values 加噪实现，
与 TF 的 ``sparse_retain`` + 缩放等价。
"""

# ------------------------------------------------------------------ 导入依赖
import numpy as np                  # 仅用于 np.sqrt 计算 Glorot 初始化范围
import torch                         # 张量与稀疏算子（torch.mm / torch.sparse.mm）
import torch.nn as nn                # 继承 Module/Parameter/ParameterList
import torch.nn.functional as F      # relu / dropout 等函数式 API


# =========================================================== 稀疏 dropout
def sparse_dropout(x, keep_prob):
    """稀疏张量的 dropout（等价于 TF 版的 ``sparse_retain``）。

    TF 原版（严格等价移植）：
        random_tensor = keep_prob + tf.random_uniform(noise_shape)
        mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        out = tf.sparse_retain(x, mask) * (1 / keep_prob)

    注意：``floor(keep_prob + U(0,1))`` 的保留概率**恰好是 keep_prob**
    （当 keep_prob<1 时 floor 结果为 0 或 1），但用的是均匀分布而非伯努利分布，
    必须照搬才能与 TF 版逐位对齐。

    另一个关键点：噪声只作用在**非零元素**上。若对整个稠密矩阵做 dropout，
    原本为 0 的位置会被 U(0,1)/keep_prob 污染成极小的非零值，
    既破坏稀疏结构也会改变图传播的语义。

    :param x: torch 稀疏张量（coalesce 后的 ``sparse_coo_tensor``）
    :param keep_prob: 保留概率，取值 [0, 1]
    :return: 与 x 结构相同、values 被随机置零并缩放的稀疏张量
    """
    # 快速路径：keep_prob=1 表示不 dropout，原样返回（避免无谓的随机数消耗，
    # 保证与 TF 版在推理阶段的随机数序列消耗一致）
    if keep_prob >= 1.0:
        return x

    # ---- 拆出稀疏张量的三要素：values（非零值）/ indices（坐标）/ shape ----
    # 只对 values 加噪，indices 与 shape 完全不动 —— 这保证了稀疏结构不变。
    values = x._values()             # 形状 (nnz,)，即所有非零元素的取值
    indices = x._indices()           # 形状 (2, nnz)，即每个非零元素的 (行, 列)
    shape = x.shape                   # 形状 (N, N)，稀疏矩阵的完整维度

    # ---- 构造 dropout 掩码 ----
    # U(0,1) + keep_prob 的取值范围是 (keep_prob, keep_prob+1)，
    # 对 keep_prob<1 取 floor，结果只能是 0 或 1，故天然是布尔掩码。
    random_tensor = keep_prob + torch.rand_like(values)   # (nnz,)
    dropout_mask = torch.floor(random_tensor)             # (nnz,)，取值 0/1

    # ---- 应用掩码并做「 inverted dropout」缩放 ----
    # 除以 keep_prob 是为了保持期望不变：E[mask/keep_prob] = keep_prob/keep_prob = 1
    # 因此训练时的输出期望与推理时一致，不需要额外做 BatchNorm 之类的修正。
    # coalesce() 会重新排序并合并重复坐标，保证后续 torch.sparse.mm 高效执行。
    return torch.sparse_coo_tensor(
        indices, values * dropout_mask / keep_prob, shape).coalesce()


# =========================================================== 图卷积层
class GraphConvolution(nn.Module):
    """图卷积层。

    对应文档 §3.3 的传播规则：
        H^(l+1) = σ( D̃^-1/2 Ã D̃^-1/2 H^(l) W^(l) )

    支持多个 support（如 Chebyshev 基），每个 support 拥有独立权重矩阵，
    最终输出求和 —— 这正是 ChebNet 到 GCN 的推广形式：

        H^(l+1) = σ( (Σ_{i=0..K} Ã_i H^(l) W_i^(l)) + b )

    其中当 K=0 且 Ã_0 = D̃^-1/2 Ã D̃^-1/2 时，式子退化为标准 GCN。
    """

    def __init__(self, input_dim, output_dim, supports, dropout=0.,
                 act=F.relu, bias=False, sparse_inputs=False,
                 featureless=False, logging=False, name=None):
        """构造图卷积层并初始化参数。

        :param input_dim:  输入特征维度 F_in
        :param output_dim: 输出特征维度 F_out
        :param supports:   support 稀疏矩阵列表，长度即 K+1
        :param dropout:     丢弃概率 p（注意是丢弃率，保留率为 1-p）
        :param act:         激活函数；输出层传 ``identity`` 以保留 logits
        :param bias:        是否使用偏置项
        :param sparse_inputs: 输入是否为稀疏张量（第一层为 True）
        :param featureless: 输入是否已被 support 传播过（避免重复传播）
        :param logging:     是否打印调试信息（对齐 TF 版的 logging 开关）
        :param name:        层名，便于与 TF 版逐层对照
        """
        super(GraphConvolution, self).__init__()   # 初始化 nn.Module 内部状态

        # ---- 逐个保存超参到实例属性（forward 中要用）----
        self.supports = supports             # support 稀疏矩阵列表
        self.act = act                       # 激活函数
        self.sparse_inputs = sparse_inputs   # 输入是否稀疏
        self.featureless = featureless       # 是否跳过输入侧的线性变换
        self.bias = bias                     # 是否加偏置
        self.dropout = dropout               # 丢弃概率
        self.logging = logging               # 调试开关
        self.name = name                     # 层名

        # 每个 support 一个权重矩阵（对应 TF 的 weights_0, weights_1, ...）
        self.weights = nn.ParameterList()    # ParameterList 才能被 optimizer 识别
        for i in range(len(self.supports)):  # 遍历 support 个数 = K+1
            # torch.empty 只分配内存不初始化，随后立刻调 reset_parameters 填值
            self.weights.append(nn.Parameter(torch.empty(input_dim, output_dim)))
            # Glorot 初始化，范围依赖本层的输入/输出维度
            self.reset_parameters(self.weights[i], input_dim, output_dim)

        if self.bias:                        # 需要偏置时分配全零参数
            self.bias_param = nn.Parameter(torch.zeros(output_dim))
        else:
            # 不用偏置时必须显式注册为 None，否则 optimizer 遍历 state_dict
            # 时会因为找不到该键而报 KeyError
            self.register_parameter('bias_param', None)

    @staticmethod
    def reset_parameters(param, input_dim, output_dim):
        """Glorot & Bengio (AISTATS 2010) 初始化，与 TF 原版 glorot() 一致。

        采样范围 ±sqrt(6/(F_in+F_out))，使前向输出的方差约为 1/3，
        配合正交化的输入分布可获得最陡的梯度下降曲线。
        """
        init_range = np.sqrt(6.0 / (input_dim + output_dim))  # Glorot 上界
        # no_grad：初始化不应被 autograd 记录，否则会产生无意义的计算图节点
        with torch.no_grad():
            param.uniform_(-init_range, init_range)           # 均匀分布填充

    def forward(self, inputs):
        """前向传播：dropout -> 按 support 线性变换 -> support 传播 -> 求和 -> 激活。

        :param inputs: (N, F_in) 的稀疏或稠密张量
        :return: (N, F_out) 的激活后输出
        """
        x = inputs                          # 局部变量，避免覆盖入参

        # ---- 1. dropout ----
        # 稀疏输入必须走 sparse_dropout，否则零位置会被污染成非零
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout)   # 传的是保留率
        else:
            # 稠密路径：F.dropout 内部在 eval 模式下自动跳过
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- 2. convolve：对每个 support 做 (S_i · X · W_i) 再求和 ----
        supports = []                       # 收集每个 support 分支的输出
        for i in range(len(self.supports)): # 遍历 K+1 个 support
            if not self.featureless:
                # 先做特征侧线性变换 X·W_i；稀疏时必须用 torch.sparse.mm，
                # 它会自动选择稀疏×稀疏或稀疏×稠密的高效内核
                pre_sup = torch.sparse.mm(x, self.weights[i]) if self.sparse_inputs \
                    else torch.mm(x, self.weights[i])
            else:
                # featureless=True 表示 x 已被 support 传播过（形状已等于 F_out），
                # 无需再乘权重矩阵，直接把权重本身当作待传播的对象
                pre_sup = self.weights[i]

            # support 前向：稀疏矩阵 × 稠密矩阵，(N,N)·(N,F_out) -> (N,F_out)
            support = torch.sparse.mm(self.supports[i], pre_sup)
            supports.append(support)         # 收集本分支结果

        # stack 得到 (K+1, N, F_out)，sum(dim=0) 沿 support 维求和 -> (N, F_out)
        output = torch.sum(torch.stack(supports), dim=0)

        # ---- 3. 偏置 ----
        # 广播加法：(N,F_out) + (F_out,) -> (N,F_out)
        if self.bias:
            output = output + self.bias_param

        # ---- 4. 激活 ----
        # 输出层传 identity，保留原始 logits 供 softmax 交叉熵使用
        return self.act(output)


# =========================================================== 全连接层
class Dense(nn.Module):
    """全连接层（MLP 基线用）。

    与 GraphConvolution 的唯一区别是**没有 support 传播**，
    即少了那个 ``S_i ·`` 的步骤，其余（dropout / 权重 / 偏置 / 激活）完全一致。

    保留这个类是为了让 ``--model dense`` 的对照组与 GCN 组共享同一套
    参数初始化与损失代码，从而使对比实验只差「有没有图信息」这一个变量。
    """

    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=F.relu, bias=False, sparse_inputs=False,
                 featureless=False, logging=False, name=None):
        """构造全连接层并初始化参数（参数含义与 GraphConvolution 相同）。"""
        super(Dense, self).__init__()        # 初始化 nn.Module 内部状态

        # ---- 逐个保存超参到实例属性（forward 中要用）----
        self.act = act                       # 激活函数
        self.sparse_inputs = sparse_inputs   # 输入是否稀疏
        self.featureless = featureless       # 兼容参数（本层不使用）
        self.bias = bias                     # 是否加偏置
        self.dropout = dropout               # 丢弃概率
        self.logging = logging               # 调试开关
        self.name = name                     # 层名

        # 全连接层只有一个权重矩阵 (F_in, F_out)，直接注册为 Parameter
        self.weights = nn.Parameter(torch.empty(input_dim, output_dim))
        # Glorot 初始化
        self.reset_parameters(self.weights, input_dim, output_dim)

        if self.bias:                        # 需要偏置时分配全零参数
            self.bias_param = nn.Parameter(torch.zeros(output_dim))
        else:
            # 显式注册为 None，保持 state_dict 结构与 GraphConvolution 一致
            self.register_parameter('bias_param', None)

    @staticmethod
    def reset_parameters(param, input_dim, output_dim):
        """Glorot 初始化，公式与 GraphConvolution.reset_parameters 相同。"""
        init_range = np.sqrt(6.0 / (input_dim + output_dim))  # Glorot 上界
        with torch.no_grad():               # 初始化不进计算图
            param.uniform_(-init_range, init_range)           # 均匀分布填充

    def forward(self, inputs):
        """前向传播：dropout -> 线性变换 -> (+bias) -> 激活。

        :param inputs: (N, F_in) 的稀疏或稠密张量
        :return: (N, F_out) 的激活后输出
        """
        x = inputs                          # 局部变量，避免覆盖入参

        # ---- 1. dropout（与图卷积层完全相同的处理）----
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout)   # 传保留率
        else:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- 2. 线性变换 X·W，无 support 传播 ----
        output = torch.sparse.mm(x, self.weights) if self.sparse_inputs \
            else torch.mm(x, self.weights)

        # ---- 3. 偏置 ----
        if self.bias:                        # 广播加法 (N,F_out) + (F_out,)
            output = output + self.bias_param

        # ---- 4. 激活 ----
        return self.act(output)              # 输出层传 identity 保留 logits
