# `Keras` 源码分析

*此文档中，凡代码里用`pass`，均系省略源码以便阅读，起“本枝百世”之用。此注明者，乃`pass`非源码所有，勿叫读者疑心不解也。*

[TOC]

## `Keras` 概览

我们从一个简单的全连接分类器来看`Keras`的设计原则和阅读源代码。在`Keras`的官网上有这样一个简单全连接网络的示例[The Sequential model API](https://keras.io/models/sequential/)：

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras import backend as K

    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    model.fit(train['data'], train['label'], batch_size=32, 
          nb_epoch=10, verbose=1)

其中，`Sequential`模型的代码在`keras/models.py`中。
后端backend的代码在`keras/backend`里。
网络的核心概念——层(`Layer`)的核心源码则在`keras/engine/topology.py`中，`Dense`网络只是`Layer`类的一个继承，其他其他所有的层都是这样的一种继承，所以developer可以通过继承`Layer`类来实现自己需要的层。

从整体上看，`Keras`源码的组织和功能是这样的：

    .
    │  activations.py
    │  callbacks.py
    │  constraints.py
    │  initializations.py
    │  metrics.py
    │  models.py
    │  objectives.py
    │  optimizers.py
    │  regularizers.py
    │  __init__.py
    │  
    ├─applications
    |      # 一些典型的应用
    │      ...
    │      
    ├─backend
    |      # Theano, Tensorflow 后端
    |      # tensorflow_backend.py 和 theano_backend.py 有一些同名的函数
    |      # 这样 import backend as K 以后应用时，就不需要考虑 Tensorflow 和 Theano 的具体差别了
    │      common.py
    │      tensorflow_backend.py
    │      theano_backend.py
    │      __init__.py
    │      
    ├─datasets
    |      # 下载数据集的脚本
    │      ...
    |
    ├─engine
    │      topology.py # Keras Layer, Input, Merge, Container的基础
    │      training.py
    │      __init__.py
    │      
    ├─layers
    |      # 相当于 engine 的应用
    |      # 通过继承 engine/topology.py 的 Layer 来实现不同的层
    │      convolutional.py
    │      __init__.py
    │      ...
    │      
    ├─preprocessing
    │      image.py
    │      sequence.py
    │      text.py
    │      __init__.py
    │      
    ├─utils
    │      data_utils.py
    │      generic_utils.py
    │      io_utils.py
    │      layer_utils.py
    │      np_utils.py
    │      test_utils.py
    │      visualize_util.py
    │      __init__.py
    │      
    └─wrappers
            scikit_learn.py
            __init__.py


## backend 设计

首先要说一下后端设计。`Keras`最初后端只有`Theano`，现在可以支持`Tensorflow`。
`Keras`之所以易于扩展backend，是因为*后端采用的函数名都一样*。这等于说是在`Tensorflow`和`Theano`基础上*又向上封装了一层*。

在`backend/theano_backend.py`和`tensorflow_backend.py`两个文件中，封装到backend中的函数有：[Backend functions](https://keras.io/backend/)

这些同名函数的功能基本上如字面意思所述，例如：

    def maximum(x, y):
        return tf.maximum(x, y)
    def maximum(x, y):
        return T.maximum(x, y)

`Tensorflow`和`Theano`的差别基本就如上例所示。由于存在大量同名的函数，所以在调用后端时只需要：

    import backend as K
    K.function_name(args)

我们可以通过这些重名的函数看到，哪些构件对于深度学习而言是基本的、必需的，这对于芯片设计会有一定的启发。

## `class Layer(object)` 设计

`class Sequential(Model)`继承了`keras/engine/training.py`中的`Model`类，而`Model`类则继承了同目录下的`keras/engine/topology.py`中的`Container`类，`Container`类继承了同文件中的`Layer`类。

也就是说，`Sequential`模型实际上是泛型模型([functional API](https://keras.io/models/model/))的一个特殊情况。而泛型模型又是容器(`Container`)、是层(`Layer`)的特殊情况，因此有必要先搞清楚`Layer`的设计原理。

### `class Node(object)`

`Layer`类和`Node`类很有关系。两个`Layer`之间用`Node`连接。`Layer`有`inbound_nodes`和`outbound_nodes`两种`list`，他们的元素都是`Node`，用来绑定输入与输出的张量。

每当一个`Layer`接收新的输入张量时，就在`layer.inbound_nodes`中增加一个`Node`。同理，当一个输出张量被另一层`Layer`调用时，在`layer.outbound_nodes`增加新的节点。`Node`的作用类似于*函数之间的参数传递*。

    class Node(object):
        def __init__(self, outbound_layer,
                     inbound_layers, node_indices, tensor_indices,
                     input_tensors, output_tensors,
                     input_masks, output_masks,
                     input_shapes, output_shapes):
            '''
            构造函数
            outbound_layer 
                此 Node 绑定的输出 Layer ，也就是说当前 Node 在 outbound_layer 的 inbound_nodes 中；
            inbound_layers
                输入 Layer，当前 Node 作为其 outbound_nodes 的元素

            下面的循环将 Node 加入到所有要绑定的输入 Layer 中。
            同时，也绑定了要输出的 Layer 的 Node。
            '''
            for layer in inbound_layers:
                if layer is not None:
                    layer.outbound_nodes.append(self)
            outbound_layer.inbound_nodes.append(self)

        @classmethod # 指定函数 create_node 为类方法而非实例方法，因此可以直接进行类调用 Node.create_node()

        def create_node(cls, outbound_layer,
                        inbound_layers, node_indices=None, tensor_indices=None):
            '''
            inbound_layers
                从 inbound_layers.inbound_nodes 中读取所有的输入 Node 信息，包括数据、mask、shape。
            outbound_layer
                根据从 inbound_layers 读到的足够多的信息来确定新建一个 Node 传递给 outbound_layers 。
            函数返还一个outbound_layers 的 Node 类节点。
            '''
            return cls(outbound_layer,
                       inbound_layers, node_indices, tensor_indices,
                       input_tensors, output_tensors,
                       input_masks, output_masks,
                       input_shapes, output_shapes)

        def get_config(self):
            '''
            返还输入输出层的名字、节点与张量的索引
            '''
            return {'outbound_layer': self.outbound_layer.name if self.outbound_layer else None,
                    'inbound_layers': inbound_names,
                    'node_indices': self.node_indices,
                    'tensor_indices': self.tensor_indices}


`Node`和`Layer`互为成员变量，所以在`Layer`创建的时候就已经创建了，不需要单独创建。

### `class Layer(object)`

至于`Layer`类，它主要包括这些成员变量(properties)、实例方法(methods)和类方法(class methods)

#### 主要的 Properties

- `input_spec`

    `class InputSpec`的`list`。每一个元素描述了对于输入的要求，例如维度`ndim`和数据类型`dtype`。

- `trainable`

    用来标志这个`Layer`在训练中权重是否更新(训练)的`bool`值

- `input_shape, output_shape`

- `inbound_nodes`, `outbound_nodes`

    `Layer`之间存放的`Node`的`list`

- `input, output`

    输入输出的张量（`tensor`）

- `trainable_weights, non_trainable_weights, weights`
    可以训练、不可以训练的变量`list`，`weights`是他们的串接。它们是以函数形式存在的property，返回`list`。

#### 关键的 Methods

- `I/O`相关的 Methods
    - `def create_input_layer(self, batch_input_shape, input_dtype=None, name=None)`
    
        这个函数会按照输入参数修改当前`Layer`的`batch_input_shape`和`input_dtype`，并且调用函数`def Input(shape=None, batch_shape=None, name=None, dtype=K.floatx(), sparse=False, tensor=None)`得到一个`Keras tensor`，`x`。

		    x = Input(batch_shape=batch_input_shape,
		      dtype=input_dtype, name=name)
		    self(x)

        `Keras tensor`是在backend的 tensor 基础之上增加内容的张量。用返还的`Keras tensor`将自身实例化为`Layer`，这是为了创造当前`Layer`与刚刚创造的输入`Layer`之间的连接`Node`。
        `Keras tensor`实际上是`InputLayer`输入`Node`的输出张量：

            input_layer = InputLayer(batch_input_shape=batch_shape,
                                     name=name, input_dtype=dtype,
                                     sparse=sparse,
                                     input_tensor=tensor)
            outputs = input_layer.inbound_nodes[0].output_tensors
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs

        因为在子类`InputLayer`中并没有调用该函数，所以没有矛盾的地方。

- 与`Losses`, `Update`有关的 Methods

    - `def add_loss(self, losses, inputs=None)`

        这个函数会不断添加`self.losses`列表，参数`losses`会被转化为`list`然后被加到`self.losses`后面。
        然后根据参数`inputs`，获得它的用户级编号`uid`作为`hash`值。`uid`是根据`python`的`id()`函数得到的，某种意义上类似于`C`的内存地址。

            inputs_hash = object_list_uid(inputs)

        然后将`losses`列表加入对应的`hash`值位置：

            self._per_input_losses[inputs_hash] += losses

    - `def get_losses_for(self, inputs)`
    
        将`add_loss`函数设定的`inputs`位置的`losses`取出来

    - `update`类的函数
    
        基本上和`losses`都一样，只是将关键字`losses`改成`updates`。

- `Weight`相关的 Methods
    - `def weights(self)`
    
        串接可训练与不可训练的权重：

            return self.trainable_weights + self.non_trainable_weights

    - `def set_weights(self, weights)`
    
        将`self.weights`和参数`weights`的张量载入到`[numpy.array]`形式的`weight_value_tuples`

    -  `def get_weights(self)`
    
        以`[numpy.array]`的形式返回当前`Layer`的张量

### 以 `Dense` 层为例

`Dense`层是`Keras`中最简单的一个全连接的网络。整个`Dense`层的代码大致如下：

    class Dense(Layer):
        def __init__(self, output_dim, init='glorot_uniform',
                     activation=None, weights=None,
                     W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                     W_constraint=None, b_constraint=None,
                     bias=True, input_dim=None, **kwargs):
            pass
            super(Dense, self).__init__(**kwargs)

        def build(self, input_shape):
            pass

        def call(self, x, mask=None):
            output = K.dot(x, self.W)
            if self.bias:
                output += self.b
            return self.activation(output)

        def get_output_shape_for(self, input_shape):
            pass

        def get_config(self):
            pass

首先，`Dense`是对父类`Layer`的继承，但是覆盖了

- `build`
    定义权重。可以训练的加入`self.trainable_weights`，不可以训练的加入`self.non_trainabe_weights`，需要更新的以`(old_tensor, new_tensor)`的形式加入`self.updates`。
- `call`
    定义功能，具体的数学运算。
- `get_output_shape_for`
    给`Keras`指明shape的变化。
- `get_config`
    给出`Layer`的确认信息，包括`output_dim`, `W_constraint`等。

这四个函数，这四个函数具有“多态”的特点。

在`Dense`实例化时，在构造函数`__init__`的结尾调用父类`Layer`的构造函数，这时候`Layer`调用的多态函数就被子类覆盖了，实现了子类的特有功能。

在官方手册[Writing your own Keras layers](https://keras.io/layers/writing-your-own-keras-layers/)中，并不需要用户实现`get_config`，只需要自己编写另外三个多态函数即可。

### 数学运算

可见`Layer`的计算功能集中在`call`函数。

`Dense`的`call`如上面的代码所示，它实际上还是按照输入的`activation`关键字调用了`keras/activations.py`中的激活函数。
`keras/activations.py`提供了如下这些激活类型，

- def softmax(x)
- def elu(x, alpha=1.0)
- def softplus(x)
- def softsign(x)
- def relu(x, alpha=0., max_value=None)
- def tanh(x)
- def sigmoid(x)
- def hard_sigmoid(x)
- def linear(x)

在不指定`activation`参数的情况下，参数传递为`None`，默认调用`linear`。

*由此可见，`Layer`只具有前向传播的计算能力，不具备反向传播的计算能力。*


## `class Container(Layer)` 设计

`Container`是由`Layer`组成的有向无环的计算图(a directed acyclic graph of layers)，实际上是一个`Model`的拓扑结构。`Container`和`Model`之间的差别在于训练，所以在构造时，`Model`是对`Container`的继承。

### `__init__()` 函数

构造函数`__init__()`通过一种“自顶向下”的方法构造了计算图模型。在`__init__()`中，首先会将参数输入的`input`, `output`这两张层的张量处理好：

    def __init__(self, input, output, name=None):
        # 先将输入的张量`input`, `output`
        # 处理成`Container`专用的`tensor list`
        # Container-specific properties.
        if isinstance(input, (list, tuple)):
            self.inputs = list(input)  # Tensor or list of tensors.
        else:
            self.inputs = [input]
        if isinstance(output, (list, tuple)):
            self.outputs = list(output)
        else:
            self.outputs = [output]

        # Build self.output_layers:
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            # 添加输出*层*
            self.output_layers.append(layer)
            # 添加输出层的*结点*
            self.output_layers_node_indices.append(node_index)
            # 添加*张量*
            self.output_layers_tensor_indices.append(tensor_index)

        # Build self.input_layers:
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            # It's supposed to be an input layer, so only one node
            # and one tensor output.
            assert node_index == 0
            assert tensor_index == 0
            self.input_layers.append(layer)
            self.input_layers_node_indices.append(node_index)
            self.input_layers_tensor_indices.append(tensor_index)

        # `output_layers`, `input_layers` 的 cache 处理

#### Graph 构建

接下来会通过下面这个`__init__`内部定义的函数来递归地构造计算图

        def build_map_of_graph(tensor, seen_nodes=set(), depth=0,
                               layer=None, node_index=None, tensor_index=None):
构造出来的计算图大致是这样的：

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

“自顶向下”的构造从`output` layer 开始，逐个检查它的`inbound_nodes`列表中的结点，将结点加入“图可见结点”(`seen_nodes`)中。因为`output` layer 到 `input` layer 之间并不是所有结点都有用的，只有在`seen_nodes`中的才是计算图模型所需要的。

最后再遍历当前结点的`inbound_layers`。对于每一个`layer`，需要继续向下添加`seen_nodes`。这样，就递归地构建了计算图。

同时，对于一个新构造的`Container`而言，作为不带有训练功能的拓扑结构，它的`inbound node`只有一个，并且没有`outbound node`。

#### Depth 环路避免

为了在有向图中防止出现环，所以采用`depth`（深度）对`Node`和`Layer`进行描述。按照`depth`的顺序，获得经过排序的`self.layers_by_depth`和`self.nodes_by_depth`。

在有向图中利用`depth`来避免出现环是很容易理解的，因为如果出现有向环的话，那么`Node`和`Layer`的深度就会不断增加以至于无穷大，也就是所谓的“无穷计数问题”。

### 有关训练的 Property Methods

#### Update

在方法`def updates(self)`中，确定某一个`Layer`是否需要更新的方法就是检查`'updates'`属性：

    @property
    def updates(self):
        updates = []
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                # 根据`layer.inbound_nodes`进行更新
                pass
        return updates

#### Loss

损失函数也一样，通过检查`'losses'`属性和检查`layer.inbound_nodes`来完成。

### Weights 相关的 Methods

    @property
    def trainable_weights(self):
        pass

    @property
    def non_trainable_weights(self):
        pass

这两个函数中的权重选取就根据`Layer.trainable`这个属性来进行选择。在`model.summary()`中，可以查看可训练与不可训练的参数数量，就是通过这两个函数实现的。

### 前向传播计算

#### `output` 计算

在这里最重要的函数是

    def run_internal_graph(self, inputs, masks=None):
        # Computes output tensors for new inputs.
        pass

因为之前在建图的时候有记录`depth`信息，所以数据的流动、计算可以逐层进行：

        depth_keys = list(self.nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = self.nodes_by_depth[depth]
            for node in nodes:
                # This is always a single layer, never a list.
                pass

因为是有向无环图，而且图又自有深度学习任务的特点，所以必然在每一层都只有一个`Layer`。这是上面这段大循环的基础，这段循环将从深`depth`到浅`depth`遍历层，也就是从`input layer`到`output layer`遍历。

对于每一个层而言，前向传播计算就是将从`input`层来的、已经算过的张量拿过来计算，并且传给下一层。具体的计算就要通过`Layer`的`call`方法实现：

    output_tensors = to_list(layer.call(computed_tensors,
                                        computed_masks))
    output_masks = to_list(layer.compute_mask(computed_tensors,
                                              computed_masks))

每完成一张层的前向计算，要添加一下`update`，`loss`以及缓冲池的记录，并且更新`_keras_shape`。这样方便将来的训练。当上面的大循环走完时，数据流也就从`input layer`流到了`output layer`。此时再将数据收集起来即可：

        for x in self.outputs:
            tensor, mask = tensor_map[str(id(x))]
            pass
            output_tensors.append(tensor)
            output_masks.append(mask)

#### Method `call` 与 cache 策略

`call`函数没有承担主要计算任务，计算任务主要还是由`run_internal_graph`方法实现的。

但是`call`利用了一个巧妙的缓冲策略降低了调用`run_internal_graph`的次数（显然，这个函数要进行一次图的全局计算，代价相对比较高）。在`Container`的构造函数`__init__`中，特别预置了三个`dict`：

        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

就是为了起到缓冲作用，降低调用`run_internal_graph`的次数。在`call`中，可以清晰地看到它们的作用：

    def call(self, input, mask=None):
        pass
        if cache_key in self._output_tensor_cache:
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, output_masks, output_shapes = self.run_internal_graph(inputs, masks)
            return output_tensors

这个缓冲的设计，或许也可以作为芯片设计的参考。

到此为止，`keras/engine/topology.py`中的主要三个类：

- `Node`
- `Layer`
- `Container`

已经解释过了，其实还有诸如`Merge`之类的内容也是很重要的。但如果只是`Sequential`模型，就未必需要在这里添加说明以起探微`Keras`代码之功效，读者可以自去读源码。

## `class Model(Container)` 设计

`Model`类相比于`Container`类而言，最大的特点就是它具有了反向传播的能力，换而言之也就是说`Model`可以进行训练，这一点落在`fit`函数上。至于其他的方法，诸如`predict`之类，对于理解深度学习框架而言并不十分重要。因此，主要需要理解的就是`compile`和`fit`两个函数，实际上用户在进行训练时，也是这两个函数最重要。

### Method `compile`

`compile`函数对输入进行了一些确认

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, **kwargs):
        pass

注意到用户在调用`compile`时，实际上还没有填充训练集与样本标签，例如我们最开始使用的例子：

    model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

#### 化归格式

`compile`会对用来设定模型的参数进行检查，包括数据类型检查等等，以化归到恰当的形式。例如`loss`可能有`dict`和`list`的表达，那么`compile`要分别处理两种表达的输入。

#### 准备预测

预测结果用`self.targets`来储存：

        self.targets = []
        for i in range(len(self.outputs)):
            shape = self.internal_output_shapes[i]
            name = self.output_names[i]
            self.targets.append(K.placeholder(ndim=len(shape),
                                name=name + '_target',
                                sparse=K.is_sparse(self.outputs[i]),
                                dtype=K.dtype(self.outputs[i])))

用`Tensorflow`和`Theano`后端产生一个占位符。

#### `loss` 计算

误差计算用函数参数`loss_weights`计算权重，加权计算总的误差：

        total_loss = None
        for i in range(len(self.outputs)):
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            weighted_loss = weighted_losses[i]
            sample_weight = sample_weights[i]
            mask = masks[i]
            loss_weight = loss_weights_list[i]
            output_loss = weighted_loss(y_true, y_pred,
                                        sample_weight, mask)
            if len(self.outputs) > 1:
                self.metrics_tensors.append(output_loss)
                self.metrics_names.append(self.output_names[i] + '_loss')
            if total_loss is None:
                total_loss = loss_weight * output_loss
            else:
                total_loss += loss_weight * output_loss

最后，对于每一组真实值与预测值，都要确定计算他们之间`loss`的目标函数：

        for i in range(len(self.outputs)):
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            output_metrics = nested_metrics[i]

            for metric in output_metrics:
                if metric == 'accuracy' or metric == 'acc':
                    output_shape = self.internal_output_shapes[i]
                    acc_fn = None
                    # 选用不同的目标函数
                    if output_shape[-1] == 1 or self.loss_functions[i] == objectives.binary_crossentropy:
                        acc_fn = metrics_module.binary_accuracy
                    elif self.loss_functions[i] == objectives.sparse_categorical_crossentropy:
                        acc_fn = metrics_module.sparse_categorical_accuracy
                    else:
                        acc_fn = metrics_module.categorical_accuracy

                    append_metric(i, 'acc', acc_fn(y_true, y_pred))
                else:
                    pass

目标函数本身是在`keras/metrics.py`中，在`keras/engine/training.py`中被调用为`metrics_module`

    from .. import metrics as metrics_module

这部分代码是用backend写成的。以(0, 1)二值标签为例，目标函数为：

    def binary_accuracy(y_true, y_pred):
        return K.mean(K.equal(y_true, K.round(y_pred)))

### Method `fit`

`fit`函数是具有批量反向传播训练能力的函数：

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0):
        pass

#### 数据标准化

在`fit`函数中，首先会调用类方法`_standardize_user_data`以进行数据处理：

        x, y, sample_weights = self._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            check_batch_dim=False,
            batch_size=batch_size)

这样得到标准化的数据。

#### 训练函数

关于验证数据`validation_data`这部分，因为有时候用户不会使用，所以就不在这里说明了。接下来就是要准备输入数据和训练函数（目标函数）：

        # prepare input arrays and training function
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        # 准备算子`self.train_function`
        self._make_train_function()
        # 给出`self.train_function`算子，供`_fit_loop`使用
        f = self.train_function

其中属性`uses_learning_phase`是从`Layer`继承来的，经过`Container`和`Model`的封装。它的本意是用来标志`Layer`是否会用到后端函数`K.in_training_phase()`或`K.in_test_phase()`。

调用`_make_train_function`以准备数据，以及准备好目标函数的算子：

    def _make_train_function(self):
        pass
        if self.train_function is None:
            # 准备`inputs`
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs = self.inputs + self.targets + self.sample_weights + [K.learning_phase()]
            else:
                inputs = self.inputs + self.targets + self.sample_weights
            # 准备`updates`
            training_updates = self.optimizer.get_updates(self._collected_trainable_weights,
                                                          self.constraints,
                                                          self.total_loss)
            updates = self.updates + training_updates
            # 调用backend，准备好目标函数的算子
            self.train_function = K.function(inputs,
                                             [self.total_loss] + self.metrics_tensors,
                                             updates=updates,
                                             **self._function_kwargs)

##### `Theano` 后端的 `Function`

后端函数`function`用传递来的参数实例化一个`Keras`的`Function`类返回：

    def function(inputs, outputs, updates=[], **kwargs):
        pass
        return Function(inputs, outputs, updates=updates, **kwargs)

`Function`类只有两个函数，除了`__init__`以外还有一个`__call__`，使其成为“可调用的类”。这相当于`Function`的对象当作函数来使用，相当于重载了括号运算符。这样就可以通过下面的代码直接求出`outputs`:

    outputs = self.train_function(ins)

`Function`的构造函数主要完成变量的更新：

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

同时，定义好了形式上的输入输出函数`self.function`。这是通过`theano.function`实现的，关于这个有用的函数可以去看`Theano`的官方手册[function - defines theano.function](http://deeplearning.net/software/theano/library/compile/function.html)

`__call__`函数的主要任务则是进行数据计算，给出`outputs`表达式的数值：

        def __call__(self, inputs):
            assert isinstance(inputs, (list, tuple))
            return self.function(*inputs)

#### `_fit_loop`

函数`fit`最终就是返回`_fit_loop`的结果，这是训练过程中的一切历史记录信息。此时原始输入的训练集已经被改造成`ins`，样本标签也成为`out_labels`，其他的参数都传递给训练函数`_fit_loop`了：

        return self._fit_loop(f, ins, out_labels=out_labels,
                              batch_size=batch_size, nb_epoch=nb_epoch,
                              verbose=verbose, callbacks=callbacks,
                              val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                              callback_metrics=callback_metrics,
                              initial_epoch=initial_epoch)

`_fit_loop`是一个抽象的函数`f(ins)`，这里`f`是从后端构建来的算子`self.train_function`，`ins`是输入的训练集。

历史记录是通过`keras/callbacks.py`搜集的：

    self.history = cbks.History()

撇开数据的处理、准备，训练的主要代码是这段循环，非常关键：

        for epoch in range(initial_epoch, nb_epoch):
            # 记录本回epoch的历史信息
            callbacks.on_epoch_begin(epoch)
            # 按照batch批次打混索引
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)
            # 得到一个批次的索引
            batches = make_batches(nb_train_sample, batch_size)
            epoch_logs = {}

以下的循环是批量对训练集进行训练，首先是准备训练集的数据切片，切片大小自然是按照批次设定的：

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(ins[-1], float):
                        # do not slice the training phase flag
                        ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                
这里调用了函数`slice_X`，这个函数是用来截取`python`的`list`和`numpy`的`array`两种格式的列表的。如此获得的`ins_batch`自然就是此回`epoch`、此`batch`批次的输入`ins`。

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
            callbacks.on_epoch_end(epoch, epoch_logs)
            if callback_model.stop_training:
                break
