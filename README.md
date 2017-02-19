#　`Keras` 源码简介

<!-- mathjax config similar to math.stackexchange -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
        inlineMath: [ ['$', '$'] ],
        displayMath: [ ['$$', '$$']],
        processEscapes: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    },
    messageStyle: "none",
    "HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"] }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

[TOC]

## 源码分析方法

谚曰:"授人以鱼不如授人以渔", 故先说分析工具. `Keras`是`Python`深度学习工具, 以`TensorFlow`或`Theano`为后端. 按照调试`Python`程序的方法, 可以用`Pdb`工具来跟踪`Keras`源码, 以资理解. 我之前调试代码的时候, 在源码上留有[注释](./keras/), 可以参考.

### 一例[(source)](./example.py)

以此简单一例[(source)](./example.py)进行源码跟踪:

    from keras.layers import Input, Dense
    from keras.models import Model
    import numpy as np

    data = np.random.normal(size=(50, 20))
    labels = np.random.randint(0, 2, (50))

    inputs = Input(shape=(20,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(1, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(data, labels)  # starts training

### `Pdb` 跟踪

在`Python3`环境下, `ubuntu`机器上, 用`Pdb`调试代码的命令如下:

    $ python3 -m pdb example.py

具体的调试命令, 其实与`Gdb`相似, 可以参考官方文档[pdb — The Python Debugger](https://docs.python.org/3.6/library/pdb.html).

### 例子中的关节处

在`example.py`上, 我们构建了一个最简单, 而且数据甚为儿戏的`Keras`泛型模型([functional API](https://keras.io/models/model/)), 其中需要跟踪`Keras`源码的有

    x = Dense(64, activation='relu')(inputs)
    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(data, labels)  # starts training

这四处. 借此可以理解:

- 网络的"层"是什么
    `keras/engine/topology.py: class Node(object)`
    `keras/engine/topology.py: class Layer(object)`
- 网络的"层"是如何搭建到一起的
    `keras/engine/topology.py: class Container(Layer)`
- 网络模型需要什么必备的元素
    `keras/engine/training.py: class Model(Container): def compile()`
- 深度学习网络是如何训练的
    `keras/engine/training.py: class Model(Container): def fit()`
    - 后端的自动求导原理

## 源码之大观

源码纷繁, 抽丝剥茧不易. 不过上例中要用到的却可以缕清层次:

    . 
    │  optimizers.py
    |  # 梯度下降之关键所在
    |  # `Keras`中反向传播的主要代码
    |  # 需要借助backend才能计算梯度
    │      
    ├─backend
    |      # 封装`Theano`, `TensorFlow`
    |      # 使得接口统一, 便于扩展后端
    │      theano_backend.py
    │      tensorflow_backend.py
    |
    ├─engine
    |      # 基本网络模型
    |      # 以基本拓扑结构为主
    |      # 不设计前向传播的计算, 也不设计反向传播的计算
    │      topology.py
    |      # 网络训练
    |      # 要通过`keras/optimizer.py`, 借用backend进行反向传播的训练
    │      training.py
    │
    └─layers
           # 在`keras/engine/topology.py`基础上
           # 定义不同的前向传播方式
           core.py

以上为至为关键处, 对理解`Keras`甚为重要, 尤其以`keras/engine/`之下的两个文件为主要, 必须要读. 此乃源码之大观也.

## 后端封装

`Keras`的后端到2017/02/18为止, 共有`Theano`, `TensorFlow`两种. 它们和`numpy`一起提供了基本数据类型, 最重要的是给出了"变量(variable)"的概念. variable对于梯度的计算而言是非常重要的, 对于这一点, `Theano`和`TensorFlow`用户应该颇有体会. 关于梯度方面将在后文提及.

`keras/backend`对两个工具进行了封装, 使得用户在使用`Keras`时可以不去考虑具体的细节. 在`keras/backend/theano_backend.py`和`keras/backend/tensorflow_backend.py`两个文件下, 存在着大量同名的函数. 例如:

    def gradients(loss, variables):
        return tf.gradients(loss, variables, colocate_gradients_with_ops=True)

    def gradients(loss, variables):
        return T.grad(loss, variables)

分别是用`TensorFlow`与`Theano`实现的自动求导的结果. 它们都对`variables`中所有的变量返还梯度:

$$ \frac{\partial{Loss}}{\partial{\mathbf{W}}}, \quad \frac{\partial{Loss}}{\partial{\mathbf{b}}} $$

而`keras/optimizer.py: class Optimizer(object): def get_gradients()`在调用backend求梯度时, 可以不去考虑两种平台具体代码的差别:

    from . import backend as K
    grads = K.gradients(loss, params)

此乃后端封装之便利也. 这也利于后端的扩展, 实际上最初`Keras`只有`Theano`为后端, 后来增加了`TensorFlow`, 所需要做的事情就是将`keras/backend/theano_backend.py`中的[这些函数](https://keras.io/backend/)用`TensorFlow`改写即可.

## 实现一个`Dense`层

实际上可以参见官方文档中关于[Writing your own Keras layers](https://keras.io/layers/writing-your-own-keras-layers/)这一章节.

### `Layer` 与 `Node`

`Node`是用来连接两个(多个)`Layer`的:

    Layer_1 --- Node_1 --- Layer_2 --- Node_2

`Layer`和`Node`都用一个`list`来管理对方, 所以会出现这样的情况:

    (Pdb) p layer.name
    'dense_3'
    (Pdb) p layer.inbound_nodes[0].inbound_layers[0].name
    'dense_2'
    (Pdb) p layer.inbound_nodes[0].inbound_layers[0].inbound_nodes[0].inbound_layers[0].name
    'dense_1'
    (Pdb) p layer.inbound_nodes[0].inbound_layers[0].inbound_nodes[0].inbound_layers[0].inbound_nodes[0].inbound_layers[0].name
    'input_1'

这是简单一例中的网络模型建立起来之后的网络情况. `'dense_3'`是最外的输出层, 所有对`'dense_3'`输入数据的`Node`被`'dense_3'`的`inbound_nodes`列表管理着. 而其中每一个`Node`的`inbound_layers`列表也管理着数据输出到`Node`的`Layer`.

使用`Node`来管理`Layer`之间的关联关系, 主要原因有:

- 用`Node`来代表网络节点, 可以方便地构建网络. `Node`可以看做网络的逻辑结点, `Layer`可以看做网络的逻辑边, 且具有前向计算能力.
- 对于要使用`Merge`的复杂情况, `Node`也会使网络更有条理. 遇上输入层, 输出层这样的特殊层, 也方便从`Layer`中扩展来.
- 可以预防网络出现有向环. 在`Container`初始化的时候, 会记录网络`Node`的深度`depth`, 从而避免产生有向环.

### 前向传播

`Layer`具有前向传播的计算能力, 这正是`Layer`最大的特点. 前向传播的计算函数是成员函数`def call()`, 下面的正是`Dense`层的`call`函数([source](./keras/layers/core.py))

    def call(self, x, mask=None):
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

在这里, 我们要关注的有三个方面:

- 权重`self.W`
    权重`self.W`是在成员函数`def build()`中实现的, 这正是`Keras`要求用户在自定义一种层时所需实现的一个多态函数, 用来覆盖`Layer.build()`.
    起初, 权重有不同的赋值方式, 例如全`0`, 全`1`, 或是服从标准正态分布. 这些不同的权重初始化方法都在[`keras/initializer.py`](./keras/initializer.py)中, 用户在定义网络的时候应当指定相应的初始化方法. 同理, 正则化方法([source](./keras/regularizers.py)), 权重限制([source](./keras/constraints.py)), 也是相似的接口.
    因此, 在`Dense`层([source](./keras/layers/core.py))中, `def build`中关于`self.W`的代码是这样的:

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

- 激活`self.activation`
    因为深度学习经常要使用激活函数:

    - softmax
    - sigmoid
    - tanh
    - ...

    因此, 这些常用的激活函数就放在[`keras/activations.py`](./keras/activations.py)中了. 用户在创建`Dense`时, 可以指明激活类型:

        x = Dense(64, activation='relu')(inputs)

    这就是用`'relu'`激活的全连接层.

- 运算`K`
    一般情况下`Python`没办法处理变量:

        >>> x = 1
        >>> y = x ** 2
        >>> y
        1
        >>> x = 2
        >>> y
        1

    在和深度学习相关的程序中, 我们有时候希望出现`y == 4`. `Python`不具备这个能力, 因此要使用`Theano`和`TensorFlow`提供"变量"这一数据类型. 也就是说, 在这里

        output = K.dot(x, self.W)

    实际上是定义了一个数学意义上的函数表达式, 一段计算过程, 而非一个数值.

于是, `def call()`实际上是前向传播的一个数学函数.

### 其他层

其他的网络层, 例如[卷积](./keras/layers/convolutional.py), [池化](./keras/layers/convolutional.py), [循环](./keras/layers/recurrent.py), [词向量](./keras/layers/embeddings.py), 其实在源码结构上其实大同小异, 主要就是:

- `def build()`
    权重的设定
- `def call()`
    前向计算过程
- `def get_output_shape_for()`
    输入数据的形状变化

这些方面有所不同.

## 网络拓扑

在例子中, 执行

    model = Model(input=inputs, output=predictions)

就会调用父类[`keras/engine/topology.py: class Container(Layer)`](./keras/engine/topology.py)的构造函数`def __init__()`构建网络模型(因为`keras/engine/training.py: class Model(Container)`没有构造函数).

类`Container`中的内容很杂, 有保存模型, 保存参数, 读取`Layer`, 等等. 这些内容其实无关宏旨, 我们需要关注的只有: 它是怎样将网络拓扑建立起来的, 也就是构造函数`def __init__()`.

### 模型构建

首先规整一下数据格式, 例如将`input`收集到一个`list`中, 等等. 完成这部分工作以后, 主要依靠递归地调用`def build_map_of_graph()`将网络模型构建起来.

所谓模型构建, 是指向模型添加真正需要的`Node`和`Layer`. 因为从输入层到输出层的连接之中, 有一些`Node`可能通往其他方向, 所以在模型中未必需要. 所有需要的`Node`被添加到`seen_nodes`:

    (Pdb) p seen_nodes
    {'2903341452-2', '2902905836-0', '2902897484-1', '3068938572-3'}
    (Pdb) p id(layer.inbound_nodes[0])
    2902905836
    (Pdb) p id(layer.inbound_nodes[0].inbound_layers[0].inbound_nodes[0])
    2902897484
    (Pdb) p id(layer.inbound_nodes[0].inbound_layers[0].inbound_nodes[0].inbound_layers[0].inbound_nodes[0])
    2903341452
    (Pdb) p id(layer.inbound_nodes[0].inbound_layers[0].inbound_nodes[0].inbound_layers[0].inbound_nodes[0].inbound_layers[0].inbound_nodes[0])
    3068938572

`seen_nodes`收集了结点`id`与深度`depth`, 这样的层次策略有利于避免出现有向环.

收集`seen_nodes`的过程大致如下：

    output layer
    |
    #------------#------------#
    |            |            |
    node   ...   node   ...   node
                              |
    #------------#------------#
    |            |            |
    layer  ...   layer  ...   layer
    |
    #------------#------------#
    |            |            |
    node   ...   node   ...   node
    ...          ...          ...
    #
    |
    input layer

"自顶向下"的构造从`output layer`开始, 逐个检查它的`inbound_nodes`列表中的结点, 将结点加入`seen_nodes`中. 

然后从再遍历当前结点的`inbound_layers`. 对于每一个`layer`, 需要继续向下添加`seen_nodes`, 直到`input layer`. 这样, 就递归地构建了计算图。

对于一个新构造的`Container`而言, 作为不带有训练功能的拓扑结构, 它的`inbound node`只有一个, 并且没有`outbound node`. 这样的情况下, 如果没有设计`Node`, 而直接用`Layer`相连, 代码就会非常丑陋.

### 计算缓冲

`Container`在构造时, 建立了一些计算缓冲池. 做缓冲的目的是为了降低计算函数`def run_internal_graph()`的调用次数. `Container`的`def run_internal_graph()`可以看做所有层`call`的串联, 完成一次完整的前向传播计算, 最主要的前向传播计算量集中在这里.

将数据输入, 经过种种层, 最终得到输出结果, 这是一个很麻烦的过程: 每个层都需要调用各自的`def call()`. 如果网络很深, 或者计算过程复杂, 则需要花费很长时间.

因此, `Container`建立缓冲池保存计算结果:

        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

整个`Container`的`def call()`则这样设计命中缓冲:

        cache_key = ','.join([str(id(x)) for x in inputs])
        cache_key += '_' + ','.join([str(id(x)) for x in masks])
        if cache_key in self._output_tensor_cache:
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, output_masks, output_shapes = self.run_internal_graph(inputs, masks)
            return output_tensors

可以看到, 命中缓冲是根据输入`inputs`表元的`id`来设计句柄的. 因此, 从第二轮开始的epoch中, `call`可以绕开`run_internal_graph`, 直接返还缓冲池中的结果.

注意, 虽然绕开了`run_internal_graph`, 但是由于权重等数据类型都是backend变量, 所以计算结果数值会发生变化, 这样的才能起到训练的功效.

### 网络前向传播

网络的前向传播调用是这样的:

- `keras/engine/topology.py: class Container(Layer)`
    - `def __call__()`
        - `def call()`
            - `def run_internal_graph()`
            - `_output_tensor_cache`

在`call`内, 使用上一节所说的计算缓冲策略(这在本质上是依赖于backend提供的数据类型的). 真实进行前向传播计算的是`run_internal_graph`([source](./keras/engine/topology.py)), 它的主干代码大致如下(经过改动):

    tensor_map = {}
    for x, y, mask in zip(self.inputs, inputs, masks):
        tensor_map[str(id(x))] = (y, mask)

    for depth in depth_keys:
        nodes = self.nodes_by_depth[depth]
        for node in nodes:
            layer = node.outbound_layer
            for x in node.input_tensors:
                if str(id(x)) in tensor_map:
                    computed_data.append(tensor_map[str(id(x))])             
            computed_tensor, computed_mask = computed_data[0]
            output_tensors = to_list(layer.call(computed_tensor,
                                                computed_mask))

在`tensor_map`中的`Node`是前向传播的"急先锋": 在他们之前的网络都已经计算过了, 在他们之后的则还没有计算.

那么这段代码的逻辑就很清晰了: 按照层次遍历计算图中的`Node`, 对于`Node`又遍历它的结点输入张量, 如果前向传播已经到了这个张量`x`(也就是说张量`x`在`tensor_map`中), 那么就将`x`之前已经计算过的张量收集起来. 

遍历一个`Node`的结点张量, 收集computed信息之后, 调用这层网络(`node.outbound_layer`)的`call`, 计算前向传播到这里的输出结果.

## 编译

[example.py](./example.py)中, 到:

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

将进入[`keras/engine/training.py: class Model(Container): def compile()`](./keras/engine/training.py)编译网络. 

所谓"编译", 实际上是指模型的配置, 例如[优化器optimizer](./keras/optimizers.py), [目标函数objective](./keras/objectives.py), [模型评价方法metrics](./keras/metrics.py)等等. 其中值得一提的是损失(目标)函数`self.loss_functions`.

### 目标函数

形参`loss='binary_crossentropy'`传入以后, 

    loss_functions = []
    for name in self.output_names:
        loss_functions.append(objectives.get(loss[name]))
    self.loss_functions = loss_functions

通过以上代码获取[keras/objectives.py](./keras/objectives.py)中的损失函数. 此时调试就可以看到:

    (Pdb) p self.loss_functions
    [<function binary_crossentropy at 0xad55d584>]

用不同的误差函数会得到对结果的不同误差, 这些误差可以加权取数学期望, 从而得到一个整体上的综合误差. 所以, `loss`也需要`weight`:

    weighted_losses = [weighted_objective(fn) for fn in loss_functions]
    output_loss = weighted_loss(y_true, y_pred,
                                sample_weight, mask)

这样, 就将`self.loss_functions`列表中的所有`loss`结合到一起, 可以计算`y_true`, `y_pred`之间的综合误差了.

## 反向传播训练

在用户端, 训练的代码只有一句:

    model.fit(data, labels)

此调用在`keras/engine/training.py: class Model(Container)`之内, 关键的函数层次是:

- `def fit()`
    - `def _make_train_function()`
        设置训练函数, 也就是设置好误差计算, 梯度计算, 权重更新
    - `def _fit_loop()`
        真实训练, 对训练过程进行记录

### `def _make_train_fuction()`

`def fit()`中, 设置训练函数的代码如下:

    self._make_train_function()
    f = self.train_function

可以认为`f`是一个"算子", 或者说"函数空间的映射". 它将改变权重, 将网络模型的前向传播函数映射到另一个权重的前向传播函数, 并且返回误差计算, 模型评价(精度, 等等)的结果.

在`_fit_loop()`中是这样使用的:

    outs = f(ins_batch)

`_make_train_function`很简短. 首先它将整理数据, 将输入数据整理成如下形式:

    (Pdb) p inputs
    [input_1, dense_3_target, dense_3_sample_weights]

这就是前向传播函数的输入层与输出层. 前向传播向`input_1`输入数据, `dense_3`输出预测结果. 

`self.optimizer.get_updates()`将返还`(v, nv)`的新旧值更新元组列表, 这是构建好的梯度的形式计算, 或者说梯度的函数:

    training_updates = self.optimizer.get_updates(self._collected_trainable_weights,
                                                  self.constraints,
                                                  self.total_loss)
    updates = self.updates + training_updates

最终返还一个后端函数:

    self.train_function = K.function(inputs,
                                     [self.total_loss] + self.metrics_tensors,
                                     updates=updates,
                                     **self._function_kwargs)

每当调用返还的backend算子时, 会计算误差`loss`与评价`metrics`, 同时对所有权重进行更新. 重申一遍, 这就是:

    outs = f(ins_batch)

起到的效果.

#### 梯度计算

梯度计算的调用层次是这样的:

- `keras/engine/training.py: class Model(Container): def _make_train_function()`
    获得更新元组列表:

        training_updates = self.optimizer.get_updates(self._collected_trainable_weights,
                                                      self.constraints,
                                                      self.total_loss)

    - `keras/optimizers.py: class RMSprop(Optimizer): def get_updates()`
        根据梯度计算权重的更新, 新的权重值:

            def get_updates(self, params, constraints, loss):
                grads = self.get_gradients(loss, params)
                for ... :
                    self.updates.append(K.update(a, new_a))
                return self.updates

        其中, 后端的`update`函数是这样的:

            def update(x, new_x):
                return (x, new_x)

        - `keras/optimizers.py: class Optimizer(object): def get_gradients()`
            使用封装好的后端计算梯度:

                def get_gradients(self, loss, params):
                    grads = K.gradients(loss, params)
                    return grads

            可以检验这里的形参:

                (Pdb) p loss
                Elemwise{mul,no_inplace}.0
                (Pdb) p params
                [dense_1_W, dense_1_b, dense_2_W, dense_2_b, dense_3_W, dense_3_b]

            - `keras/backend/theano_backend.py: def gradients()`
                调用`Theano`计算梯度:

                    def gradients(loss, variables):
                        return T.grad(loss, variables)
            
            - `keras/backend/tensorflow_backend.py: def gradients()`
                调用`TensorFlow`计算梯度:

                    def gradients(loss, variables):
                        return tf.gradients(loss, variables, colocate_gradients_with_ops=True)

#### 权重更新

权重更新的计算方法已经由上一节给出, 具体数值的更新要依赖于下面的算子:

    self.train_function = K.function(inputs,
                                     [self.total_loss] + self.metrics_tensors,
                                     updates=updates,
                                     **self._function_kwargs)

所得到的是一个后端`Function`实例. 后端`function`函数是这样的:

    def function(inputs, outputs, updates=[], **kwargs):
        return Function(inputs, outputs, updates=updates, **kwargs)

它返还的`Function`实例在初始化的时候就会进行权重更新:

    class Function(object):
        def __init__(self, inputs, outputs, updates=[], **kwargs):
            unique_variables_to_update = {}
            for v, nv in updates:
                if v not in unique_variables_to_update:
                    unique_variables_to_update[v] = nv
            updates = unique_variables_to_update.items()
            self.function = theano.function(inputs, outputs, updates=updates,
                                            allow_input_downcast=True,
                                            on_unused_input='ignore',
                                            **kwargs)

可以检测一下`(v, nv)`更新元组:

    (Pdb) p v._keras_shape
    (20, 64)
    (Pdb) p v.get_value()
    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           ..., 
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32)

注意, 在`for v, nv in updates`循环中, `v`是theano中`shared`类型的变量(所以可以调用`get_value()`). 循环结束以后, `updates`中原来的`v`已经被`nv`替代了, 所以权重已经更新完毕.

到此, 再实例化一个`theano.function()`, 就是改变了权重关系的函数:

    self.function = theano.function(inputs, outputs, updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    **kwargs)

或者说, 原先的关系是这样的:

$$ \mathbf{outputs} = f(\mathbf{inputs}, \mathbf{W_{\text{before updating}}}) $$

现在已经变成

$$ \mathbf{outputs} = f(\mathbf{inputs}, \mathbf{W_{\text{after updating}}}) $$

这个类似于theano中的累积, 可以通过一个官网上的例子[Using Shared Variables](http://deeplearning.net/software/theano/tutorial/examples.html#basictutexamples)帮助理解.

这里的参数`outputs`, 在`__call__`:

    def __call__(self, inputs):
        return self.function(*inputs)

将`Function`当做函数来进行计算时, 会调用`Container`的`__call__`进行前向计算.

### `def _fit_loop()`

这个函数主要就是按照`epoch`, `batch`进行训练, 同时用[keras/callbacks.py](./keras/callbacks.py)对训练历史进行记录. 代码的主要结构是这样的:

    callbacks.on_train_begin()
    for epoch in range(initial_epoch, nb_epoch):
        callbacks.on_epoch_begin(epoch)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            callbacks.on_batch_begin(batch_index, batch_logs)
            outs = f(ins_batch)
            callbacks.on_batch_end(batch_index, batch_logs)
        callbacks.on_epoch_end(epoch, epoch_logs)
    callbacks.on_train_end()

几个`callbacks.on_balabala_end()`函数可以在屏幕上打印训练过程的信息. 这个函数最主要的就是使用上一节`def _make_train_function()`设置的训练函数:

    outs = f(ins_batch)

## Backend

到这里为止, 关于`Keras`本身的代码整体结构其实差不多就已经清楚了. 但是还有一个非常关键的谜题, 就是后端数学计算的根基: 

- 后端的数据结构是怎样组织的
- 如何实现符号求导

前一个问题贯穿整个`Keras`源码, 后一个问题其实也依赖于第一个, 但是由于它对深度学习意义非凡所特别提出来说.

要回答这两个问题, 必须理解`Theano`或`TensorFlow`的原理.


### 数据类型
### 符号求导