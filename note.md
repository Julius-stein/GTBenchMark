## graph Transformer 笔记

### todo
- 更多的模型
- 冗余相关
- 规范config传入方式
- 统一数据格式，研究更高效的数据sample和batch方式

### ANS_GT
数据预处理代码将PyG的Data转成dense tensor打包。相当于把node-level任务变成graph-level任务。模型仅采用Proximity作为encode。对输入embedding无影响，仅施加attn_bias。
ANS-GT模型基于Gophormer实现。由于PyG的特性，virtual global node在模型中添加而不是在数据中添加。virtual global node的proximity由预处理后的attn_bias经过cat得到。

### Gophormer
源码为dgl。我的实现方式和原来的实现方式有所不同：
- 数据处理。参考ANS-GT的实现，我将proximity作为大矩阵进行数据预处理，得到包含encode作为edge attr的Data，然后sample出子图，经过collator打包为graph level数据。（todo：由于GraphSAGE采样得到的子图大小不一，目前只简单的对数据padding填充，可能对模型性能有影响，不过Cora上测不出来）
- 模型实现。同样参考ANS-GT。使用了dense的MHA（Gophormer的模型实现基于dgl，是在GraphTransformer基础上实现的，MHA的实现是通过dgl的message passing机制实现。）
实践中注意到sparse转dense这一步的耗时很长。尝试了几种方法优化，效果一般。猜测由于多次allocate内存导致。

### GraphTransformer
源码为dgl。其模型结构为在添加LapPE和wlPE之后，直接经过若干层MHA，最后MLP读出。由于涉及稀疏注意力，原来的实现是使用dgl实现了一个graph版本的MHA，通过message passing机制实现注意力的计算。与Gophormer相同，此处同样使用pytorch官方实现的MHA来代替，使用Attention mask作为稀疏注意力的实现方式。

在计算LapPE时，出现了特征向量计算失败的问题，sp.linalg.eigs报错（scipy.sparse.linalg._eigen.arpack.arpack.ArpackError）。解决方案是如果报错了就重新计算一遍，问题解决。后续发现在为原图添加自环后也能解决该问题。

### torch.nn.MHA的问题
在debug graphtransformer的过程中，定位到使用的pytorch官方实现的MHA出现了NaN。根据研究，结论是如果使用Attention mask对Attention矩阵进行覆盖，如果某一行全部没有注意（全为True），会导致softmax出现nan。其原理和官方MHA实现的方式有关，官方MHA的实现完全是dense的，不考虑任何稀疏情景。

具体来说，官方对于Attention mask和key padding的处理是这样的：
- 将key padding mask当成B\*1\*L的tensor（原shape为B\*L），与Attention mask（shape为B\*L\*L）按行取逻辑或
- 将Attention mask处理为float格式，对于不计算注意力的位置置为-inf。
- 将Attention mask、q、k使用一个混合算子baddbmm进行q*k+bias的计算。

因此，出现nan的本质原因在于注意力矩阵的某一行全为-inf。此后，尽管这些nan数据是无效的，其理论上不会参与后续的计算，但是这些nan数据会经过一个线性层（官方实现未考虑key padding），导致模型反向传播时爆炸。

注意到单独使用key padding的时候不会出现这个问题，原因在于将key padding mask当成B\*1\*L的tensor之后直接作为Attention mask传进baddbmm中了，猜测该算子会将Attention mask先expand为B\*L\*L的矩阵再进行加法。

解决方案：在处理Attention mask的时候为padding的虚拟节点也添加自环即可。

另一个问题是torch.nn.MHA不支持添加边权作为注意力的一部分（常见做法是将边权embedding与Attention点积得到的结果进行element-wise乘法），不过目前我实现的benchmark均未考虑边权/边特征，暂时不会成为一个问题。

### 数据采样及打包
针对minibatch场景下的Graph Transformer，涉及从图中采样数据并打包的过程。对于经典的Transformer，其处理的数据主要为dense格式，而图数据在现有的框架（Pyg/DGL）中均以sparse格式存储，这就在采样时带来了额外的数据格式转换开销。对于graph-level的任务，这样的数据转换开销是可以接受的，因为可以在数据预处理阶段就完成数据的转换；而对于node-level任务，部分基于子图采样的方法需要在每个epoch都重新采样子图并转换为dense格式。

同时，针对经典Transformer的图采样也和传统的采样有些微不同：如果Transformer计算不涉及Attention bias，则只有采样出的节点特征是有用的；如果涉及Attention bias（一般Attention bias以边的特征的形式存在），则采样多hop节点时需保留整个诱导子图的边，而不仅仅是两两hop节点之间的有向边。

PyG中提供的minibatch采样数据dataloader支持子图类型设置为'induced'，但其行为在Batch size大于1时不适用于Transformer的采样逻辑。在Batch size大于1时，基于ego-graph采样的Transformer的预期行为是将多个子图分别独立进行注意力计算，并分别独立进行归一化操作；但是传统的GNN（或称为MPNN）中，这些子图可以视作一个合并的大子图，在大子图上同步进行message passing。
