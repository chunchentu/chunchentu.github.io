---
layout: post
title:  ""
date:   2017-08-19
categories: share
share: false
---


# Outline

1. Save and restore checkpoint file
2. Visualize pre-trained neural network graph from file

    2.1 Load from checkpoint file
    
    2.2 Load from protobuf file

Ref:

- [Tensorflow tutorial](https://www.tensorflow.org/programmers_guide/saved_model)

- [Turotial about saving/restoring](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)

- [A really nice jupyter notebook for loading and visualization](http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)
- [Another save/restore example](https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125)

# 1. Save and restore checkpoint file

After training your model, it would be a good idea to save your current model for the future. Or, you might start from a pre-trained model and fine-tuned it. Let's start with the regression model we have seen before. This is basically the same piece of code in the regression note. 


```python
import tensorflow as tf
import numpy as np

# number of data samples
N = 100
true_alpha = 0.3
true_beta = 0.1

# for reproducing results
np.random.seed(1234)
tf.set_random_seed(1234)
# create data for Y = 0.1X + 0.3
x_data = np.random.rand(N)
y_data = x_data*true_beta + true_alpha

# declare variables
alpha = tf.Variable(tf.zeros([1]), name="alpha")
beta = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="beta")


# relation between input and output
y_hat = x_data*beta + alpha

# loss function
loss = tf.reduce_mean(tf.square(y_hat-y_data))

# create optimizer, set step size to be 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5) 

# The object fuction is to minimize the mean squared loss
obj = optimizer.minimize(loss)

# initializer
init = tf.global_variables_initializer()
```

Before running the training step, we use **tf.train.Saver()** to manage the saving task. The next step is to train our model, as we have done before. The only difference here is that the total training step is changed from 201 to 25 so as to demonstrate how do we restore from a pre-trained model and continue training.


```python
# declare saver
saver = tf.train.Saver()

# create and run session
with tf.Session() as sess:
    sess.run(init)
    for step in range(25):
        sess.run(obj)
        print("Step:%d, alpha:%f, beta:%f" % 
              (step, sess.run(alpha), sess.run(beta)))
    # Save the variable
    save_path = saver.save(sess, "model/regressionModel.ckpt")
    print("Model saved in file: %s" % save_path)
```

    Step:0, alpha:0.043902, beta:0.578978
    Step:1, alpha:0.051716, beta:0.546352
    Step:2, alpha:0.068628, beta:0.520940
    Step:3, alpha:0.081800, beta:0.495535
    Step:4, alpha:0.094969, beta:0.472074
    Step:5, alpha:0.107131, beta:0.449887
    Step:6, alpha:0.118632, beta:0.429057
    Step:7, alpha:0.129429, beta:0.409457
    Step:8, alpha:0.139589, beta:0.391027
    Step:9, alpha:0.149142, beta:0.373694
    Step:10, alpha:0.158127, beta:0.357394
    Step:11, alpha:0.166576, beta:0.342064
    Step:12, alpha:0.174523, beta:0.327648
    Step:13, alpha:0.181996, beta:0.314090
    Step:14, alpha:0.189024, beta:0.301339
    Step:15, alpha:0.195633, beta:0.289348
    Step:16, alpha:0.201849, beta:0.278071
    Step:17, alpha:0.207695, beta:0.267466
    Step:18, alpha:0.213192, beta:0.257492
    Step:19, alpha:0.218362, beta:0.248112
    Step:20, alpha:0.223224, beta:0.239291
    Step:21, alpha:0.227797, beta:0.230995
    Step:22, alpha:0.232097, beta:0.223194
    Step:23, alpha:0.236141, beta:0.215857
    Step:24, alpha:0.239944, beta:0.208957
    Model saved in file: model/regressionModel.ckpt


After 25 training steps, we save the current training results to the file **regressionModel.ckpt** (ckpt stands for checkpoint) under **model** folder. 

Let's say you start to train the model before you sleep and then you wake up and see the output information. The estimation of $\alpha$, $\beta$ are 0.2140 and 0.2675 respectively, which may not be good enough. Fortunately, you save the variables, all you have to do is restore these variables and train the model based on the current results you have.

First we reset everything to make sure that restoring works properly. We would also generate a new set of training data just to mimic the situation that you want to further fine tune your model with the newly acquired dataset.


```python
%reset
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y


Check if variables exist.


```python
if "alpha" in locals():
    print("alpha exists")
else:
    print("alpha does not exist")
    
if "beta" in locals():
    print("beta exists")
else:
    print("beta does not exist")
```

    alpha does not exist
    beta does not exist


Actually, everything disappears after we reset the notebook. Thus, we have to redefine the neural network and other things.


```python
import tensorflow as tf
import numpy as np


# reset the tensorflow to the default graph
# this could solve some issues
tf.reset_default_graph()

###### Create a new dataset using Y = 0.1X + 0.3     ######
###### We will pretend this as newly acquired data   ######
# number of data samples
N = 100
true_alpha = 0.3
true_beta = 0.1

# for reproducing results
np.random.seed(1234)
tf.set_random_seed(1234)

x_data = np.random.rand(N)
y_data = x_data*true_beta + true_alpha
```

The next thing we want to do is to restore the model. 


```python
# declare the variables we want to restore
alpha = tf.get_variable("alpha", [1])
beta = tf.get_variable("beta", [1])
```

Note the here we declare alpha and beta using **tf.get_variable**. The is different from the previous code. We use **tf.get_variable** to create a containiner for restoring variables.


```python
# relation between input and output
y_hat = x_data*beta + alpha

# loss function
loss = tf.reduce_mean(tf.square(y_hat-y_data))

# create optimizer, set step size to be 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5) 

# The object fuction is to minimize the mean squared loss
obj = optimizer.minimize(loss)

# saver
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model/regressionModel.ckpt")
    print("Restored alpha: %s, beta: %s" % (alpha.eval(), beta.eval()) )
    
    for step in range(51):
        sess.run(obj)
        if step % 10 == 0:
            print("Step:%d, alpha:%f, beta:%f" % 
                  (step, sess.run(alpha), sess.run(beta)))
```

    INFO:tensorflow:Restoring parameters from model/regressionModel.ckpt
    Restored alpha: [ 0.23994417], beta: [ 0.20895666]
    Step:0, alpha:0.243521, beta:0.202468
    Step:10, alpha:0.269436, beta:0.155451
    Step:20, alpha:0.283460, beta:0.130008
    Step:30, alpha:0.291049, beta:0.116239
    Step:40, alpha:0.295156, beta:0.108788
    Step:50, alpha:0.297379, beta:0.104756


We could check if the restored variable **alpha** and **beta** match the previous estimation results. We also see that the second estimated values of **alpha** and **beta** are closer to the true value after 50 more training steps.

# 2. Visualize pre-trained network graph from file

Now consider another situation. Suppose we download a pre-trained model from some resources and we want to use it on our own dataset. Before restoring it, there are several things we might have interests:

- How does the network look like?

- What are the variables and their names?

- What are the inputs and outputs?


The good news is, we could use tensorboard to visualize the network. The bad news is, there are different ways to save and load the tensorflow graph ([source](https://stackoverflow.com/questions/38947658/tensorflow-saving-into-loading-a-graph-from-a-file)).

1. Load from a checkpoint file, as we obtained previously.
2. Load from a .pb ([protocol buffers](https://www.tensorflow.org/extend/tool_developers/)) file.
3. Load from models created by TF-Slim

In this note, I would show how to use the checkpoint and the .pb file. TF-Slim is a powerful and convenient library for handling complicated network models. However, it requires more understanding about its structure and we would discuss it in the later note.

Before we start, we need some useful functions. These functions come from [here](http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb), which embeds tensorboard into jupyter notebook nicely so that you don't have to open another tab in the browser like we did before. 


```python
from IPython.display import HTML

# Helper functions for TF Graph visualization

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          {% raw %}
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }} {% endraw %}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
```

## 2.1 Load from the checkpoint file

When saving the checkpoint file, you might notice that this creates several files. Among them, **XXXX.meta** stores information of the network graph and we could load the graph through function **tf.train.import_meta_graph**.


```python
tf.reset_default_graph()

_ = tf.train.import_meta_graph("model/regressionModel.ckpt.meta")

# we need graph_def for tensorboard to draw the graph
graph_def = tf.get_default_graph().as_graph_def()
show_graph(graph_def)

```
You will see something like this:

![png]({{ site.url }}/images/share/tensorflow_03_regression.png)
    


## 2.2 Load from the .pb file

See appendix to learn how to save a .pb (or .pbtxt) file. As for loading, the idea is similar to loading the checkpoint file. We first construct an empty GraphDef object and load the graph information into it. Finally, we could visualize the graph using tensorboard.


```python
tf.reset_default_graph()
graph_def = tf.GraphDef()
with open("model/CNN.pb", "rb") as f:
    graph_def.ParseFromString(f.read())
    
show_graph(graph_def)
```
You will see something like this:

![png]({{ site.url }}/images/share/tensorflow_03_CNN.png)


Next, we could import the graph using **import_graph_def** and we could print out all the operations defined in the graph.


```python
tf.import_graph_def(graph_def)
graph = tf.get_default_graph()
# this prints all the operation names
for op in graph.get_operations():
    print("{} {}".format(op.name, op.type) )
```

    import/inputs/x_input Placeholder
    import/inputs/x_img_input/shape Const
    import/inputs/x_img_input Reshape
    import/keep_prob Placeholder
    import/conv_layer_1/W_conv_1 Const
    import/conv_layer_1/W_conv_1/read Identity
    import/conv_layer_1/b_conv_1 Const
    import/conv_layer_1/b_conv_1/read Identity
    import/conv_layer_1/conv2d_1 Conv2D
    import/conv_layer_1/add Add
    import/ReLu1 Relu
    import/max_pool_2x2_1 MaxPool
    import/conv_layer_2/W_conv_2 Const
    import/conv_layer_2/W_conv_2/read Identity
    import/conv_layer_2/b_conv_2 Const
    import/conv_layer_2/b_conv_2/read Identity
    import/conv_layer_2/conv2d_2 Conv2D
    import/conv_layer_2/add Add
    import/ReLU2 Relu
    import/max_pool_2x2_2 MaxPool
    import/Reshape/shape Const
    import/Reshape Reshape
    import/fc_layer_1/W_fc_1 Const
    import/fc_layer_1/W_fc_1/read Identity
    import/fc_layer_1/b_fc_1 Const
    import/fc_layer_1/b_fc_1/read Identity
    import/fc_layer_1/MatMul MatMul
    import/fc_layer_1/add Add
    import/ReLU3 Relu
    import/dropout/Shape Shape
    import/dropout/random_uniform/min Const
    import/dropout/random_uniform/max Const
    import/dropout/random_uniform/RandomUniform RandomUniform
    import/dropout/random_uniform/sub Sub
    import/dropout/random_uniform/mul Mul
    import/dropout/random_uniform Add
    import/dropout/add Add
    import/dropout/Floor Floor
    import/dropout/div RealDiv
    import/dropout/mul Mul
    import/fc_layer_2/W_fc_2 Const
    import/fc_layer_2/W_fc_2/read Identity
    import/fc_layer_2/b_fc_2 Const
    import/fc_layer_2/b_fc_2/read Identity
    import/fc_layer_2/MatMul MatMul
    import/fc_layer_2/logits_output Add
    import/ArgMax/dimension Const
    import/ArgMax ArgMax


Now, from my understanding, the .pb file aims for production. This indicates that it is designed for conducting prediction and it cannot be fine-tuned. 

Thus, let's pretend that we have a new set of data and we would like to evaluate the model on the new dataset. From the graph we know that we are looking for **x_input** and **ArgMax** for our inputs and outputs.


```python
# pretend we get new data
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.graph_util import convert_variables_to_constants
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz



```python
x_input = graph.get_tensor_by_name("import/inputs/x_input:0")
ArgMax = graph.get_tensor_by_name("import/ArgMax:0")
keep_prob = graph.get_tensor_by_name("import/keep_prob:0")
y_input = tf.placeholder(tf.float32, [None, 10], name="y_input")

# calculate prediction accuracy
correct_prediction = tf.equal(ArgMax, 
                              tf.argmax(y_input, 1)  )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=graph) as sess:
    batch_img, batch_label = mnist.train.next_batch(100)
        
    print(sess.run(accuracy, 
                    feed_dict={x_input: mnist.test.images, 
                               y_input: mnist.test.labels,
                               keep_prob: 0.5}))
```

    0.9565


We obtain about 95% accuracy here, which is similar to the training accuracy.

## Appendix saving the CNN using .pb file


```python
%reset
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y



```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.graph_util import convert_variables_to_constants

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def weight_variable(shape, suffix=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if suffix:
        name = "W" + suffix
    else:
        name = None
        
    return tf.Variable(initial, name=name)


def bias_variable(shape, suffix=None):
    initial = tf.constant(0.1, shape=shape)

    if suffix:
        name = "b" + suffix
    else:
        name = None
        
    return tf.Variable(initial, name=name)


def conv2d(x, W, suffix=None):
    if suffix:
        name = "conv2d" + suffix
    else:
        name = None
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, suffix=None):
    if suffix:
        name = "max_pool_2x2" + suffix
    else:
        name = None
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)



# reset the tensorflow to the default graph
# this could solve some issues
tf.reset_default_graph()

# input variables
with tf.name_scope("inputs"):
    x_input = tf.placeholder(tf.float32, [None, 784], name="x_input")
    y_input = tf.placeholder(tf.float32, [None, 10], name="y_input")
    x_img_input = tf.reshape(x_input, [-1, 28, 28, 1], name="x_img_input")

# dropout setting
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# parameters for conv_1
with tf.name_scope("conv_layer_1"):
    W_conv_1 = weight_variable([5, 5, 1, 32], suffix="_conv_1")
    b_conv_1 = bias_variable([32], suffix="_conv_1")
    conv_1 = conv2d(x_img_input, W_conv_1, suffix="_1") + b_conv_1
    
ReLU_1 = tf.nn.relu(conv_1, name="ReLu1")
pool_1 = max_pool_2x2(ReLU_1, suffix="_1")

# parameters for conv_2
with tf.name_scope("conv_layer_2"):
    W_conv_2 = weight_variable([5, 5, 32, 64], suffix="_conv_2")
    b_conv_2 = bias_variable([64], suffix="_conv_2")

    conv_2 = conv2d(pool_1, W_conv_2, suffix="_2") + b_conv_2
    
ReLU_2 = tf.nn.relu(conv_2, name="ReLU2")
pool_2 = max_pool_2x2(ReLU_2, suffix="_2")

reshape = tf.reshape(pool_2, [-1, 7*7*64])

# parameters for fc_1
with tf.name_scope("fc_layer_1"):
    W_fc_1 = weight_variable([7*7*64, 1024], suffix="_fc_1")
    b_fc_1 = bias_variable([1024], suffix="_fc_1")
    fc_1 = tf.matmul(reshape, W_fc_1) + b_fc_1
    
ReLU_3 = tf.nn.relu(fc_1, name="ReLU3")
dropout = tf.nn.dropout(ReLU_3, keep_prob, name="dropout")

# parameters for fc_2
with tf.name_scope("fc_layer_2"):
    W_fc_2 = weight_variable([1024, 10], suffix="_fc_2")
    b_fc_2 = bias_variable([10], suffix="_fc_2")
    y_logits = tf.add(tf.matmul(dropout, W_fc_2), b_fc_2, name="logits_output")
    

# loss
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_input, 
                                                logits=y_logits))
# minimizaer to minimize cross_entropy
obj = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# calculate prediction accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), 
                              tf.argmax(y_input, 1)  )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


graph_def = tf.get_default_graph().as_graph_def()

# this one is tricky....
output_node_names = "ArgMax"

# start session and run

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_img, batch_label = mnist.train.next_batch(100)
        sess.run(obj, feed_dict={x_input: batch_img, 
                                 y_input: batch_label, 
                                 keep_prob: 0.5} )
        if i % 50 == 0:
            print(sess.run(accuracy, 
                           feed_dict={x_input: mnist.test.images, 
                                      y_input: mnist.test.labels,
                                      keep_prob: 0.5}))
    # save the model
    freeze_model = convert_variables_to_constants( 
                    sess,
                    graph_def,
                    output_node_names.split(","))
        
    # save in binary format
    tf.train.write_graph(freeze_model, 
                         "model", 
                         "CNN.pb", 
                         as_text=False)

    # save in txt format
    tf.train.write_graph(freeze_model, 
                         "model", 
                         "CNN.pbtxt", 
                         as_text=True)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    0.0949
    0.4977
    0.7395
    0.8148
    0.8475
    0.87
    0.8805
    0.9001
    0.9032
    0.9113
    0.915
    0.9245
    0.9272
    0.9346
    0.9359
    0.939
    0.9403
    0.9454
    0.9457
    0.9484
    INFO:tensorflow:Froze 8 variables.
    Converted 8 variables to const ops.


We save two kinds of files here. One is .pb file, which uses binary format  with smaller file size. The other is .pbtxt file. The file size would be larger but is human-readable. 

The .pb file is designed for production. This would exclude some unnecessary meta files. In tensorflow, this is called "freezing". To freeze the model, we have to convert variables to constants. Tensorflow provides an API for doing this. However, there is one tricky part: find out the **output_node_names**, which depends on your network graph structure. Normally, when doing classification, the output node should be **argmax**, **softmax** or something related to these neames.

Here is a useful code to list all of the node names:


```python
[n.name for n in tf.get_default_graph().as_graph_def().node]
```




    [u'inputs/x_input',
     u'inputs/y_input',
     u'inputs/x_img_input/shape',
     u'inputs/x_img_input',
     u'keep_prob',
     u'conv_layer_1/truncated_normal/shape',
     u'conv_layer_1/truncated_normal/mean',
     u'conv_layer_1/truncated_normal/stddev',
     u'conv_layer_1/truncated_normal/TruncatedNormal',
     u'conv_layer_1/truncated_normal/mul',
     u'conv_layer_1/truncated_normal',
     u'conv_layer_1/W_conv_1',
     u'conv_layer_1/W_conv_1/Assign',
     u'conv_layer_1/W_conv_1/read',
     u'conv_layer_1/Const',
     u'conv_layer_1/b_conv_1',
     u'conv_layer_1/b_conv_1/Assign',
     u'conv_layer_1/b_conv_1/read',
     u'conv_layer_1/conv2d_1',
     u'conv_layer_1/add',
     u'ReLu1',
     u'max_pool_2x2_1',
     u'conv_layer_2/truncated_normal/shape',
     u'conv_layer_2/truncated_normal/mean',
     u'conv_layer_2/truncated_normal/stddev',
     u'conv_layer_2/truncated_normal/TruncatedNormal',
     u'conv_layer_2/truncated_normal/mul',
     u'conv_layer_2/truncated_normal',
     u'conv_layer_2/W_conv_2',
     u'conv_layer_2/W_conv_2/Assign',
     u'conv_layer_2/W_conv_2/read',
     u'conv_layer_2/Const',
     u'conv_layer_2/b_conv_2',
     u'conv_layer_2/b_conv_2/Assign',
     u'conv_layer_2/b_conv_2/read',
     u'conv_layer_2/conv2d_2',
     u'conv_layer_2/add',
     u'ReLU2',
     u'max_pool_2x2_2',
     u'Reshape/shape',
     u'Reshape',
     u'fc_layer_1/truncated_normal/shape',
     u'fc_layer_1/truncated_normal/mean',
     u'fc_layer_1/truncated_normal/stddev',
     u'fc_layer_1/truncated_normal/TruncatedNormal',
     u'fc_layer_1/truncated_normal/mul',
     u'fc_layer_1/truncated_normal',
     u'fc_layer_1/W_fc_1',
     u'fc_layer_1/W_fc_1/Assign',
     u'fc_layer_1/W_fc_1/read',
     u'fc_layer_1/Const',
     u'fc_layer_1/b_fc_1',
     u'fc_layer_1/b_fc_1/Assign',
     u'fc_layer_1/b_fc_1/read',
     u'fc_layer_1/MatMul',
     u'fc_layer_1/add',
     u'ReLU3',
     u'dropout/Shape',
     u'dropout/random_uniform/min',
     u'dropout/random_uniform/max',
     u'dropout/random_uniform/RandomUniform',
     u'dropout/random_uniform/sub',
     u'dropout/random_uniform/mul',
     u'dropout/random_uniform',
     u'dropout/add',
     u'dropout/Floor',
     u'dropout/div',
     u'dropout/mul',
     u'fc_layer_2/truncated_normal/shape',
     u'fc_layer_2/truncated_normal/mean',
     u'fc_layer_2/truncated_normal/stddev',
     u'fc_layer_2/truncated_normal/TruncatedNormal',
     u'fc_layer_2/truncated_normal/mul',
     u'fc_layer_2/truncated_normal',
     u'fc_layer_2/W_fc_2',
     u'fc_layer_2/W_fc_2/Assign',
     u'fc_layer_2/W_fc_2/read',
     u'fc_layer_2/Const',
     u'fc_layer_2/b_fc_2',
     u'fc_layer_2/b_fc_2/Assign',
     u'fc_layer_2/b_fc_2/read',
     u'fc_layer_2/MatMul',
     u'fc_layer_2/logits_output',
     u'cross_entropy/Rank',
     u'cross_entropy/Shape',
     u'cross_entropy/Rank_1',
     u'cross_entropy/Shape_1',
     u'cross_entropy/Sub/y',
     u'cross_entropy/Sub',
     u'cross_entropy/Slice/begin',
     u'cross_entropy/Slice/size',
     u'cross_entropy/Slice',
     u'cross_entropy/concat/values_0',
     u'cross_entropy/concat/axis',
     u'cross_entropy/concat',
     u'cross_entropy/Reshape',
     u'cross_entropy/Rank_2',
     u'cross_entropy/Shape_2',
     u'cross_entropy/Sub_1/y',
     u'cross_entropy/Sub_1',
     u'cross_entropy/Slice_1/begin',
     u'cross_entropy/Slice_1/size',
     u'cross_entropy/Slice_1',
     u'cross_entropy/concat_1/values_0',
     u'cross_entropy/concat_1/axis',
     u'cross_entropy/concat_1',
     u'cross_entropy/Reshape_1',
     u'cross_entropy/SoftmaxCrossEntropyWithLogits',
     u'cross_entropy/Sub_2/y',
     u'cross_entropy/Sub_2',
     u'cross_entropy/Slice_2/begin',
     u'cross_entropy/Slice_2/size',
     u'cross_entropy/Slice_2',
     u'cross_entropy/Reshape_2',
     u'cross_entropy/Const',
     u'cross_entropy/Mean',
     u'gradients/Shape',
     u'gradients/Const',
     u'gradients/Fill',
     u'gradients/cross_entropy/Mean_grad/Reshape/shape',
     u'gradients/cross_entropy/Mean_grad/Reshape',
     u'gradients/cross_entropy/Mean_grad/Shape',
     u'gradients/cross_entropy/Mean_grad/Tile',
     u'gradients/cross_entropy/Mean_grad/Shape_1',
     u'gradients/cross_entropy/Mean_grad/Shape_2',
     u'gradients/cross_entropy/Mean_grad/Const',
     u'gradients/cross_entropy/Mean_grad/Prod',
     u'gradients/cross_entropy/Mean_grad/Const_1',
     u'gradients/cross_entropy/Mean_grad/Prod_1',
     u'gradients/cross_entropy/Mean_grad/Maximum/y',
     u'gradients/cross_entropy/Mean_grad/Maximum',
     u'gradients/cross_entropy/Mean_grad/floordiv',
     u'gradients/cross_entropy/Mean_grad/Cast',
     u'gradients/cross_entropy/Mean_grad/truediv',
     u'gradients/cross_entropy/Reshape_2_grad/Shape',
     u'gradients/cross_entropy/Reshape_2_grad/Reshape',
     u'gradients/zeros_like',
     u'gradients/cross_entropy/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim',
     u'gradients/cross_entropy/SoftmaxCrossEntropyWithLogits_grad/ExpandDims',
     u'gradients/cross_entropy/SoftmaxCrossEntropyWithLogits_grad/mul',
     u'gradients/cross_entropy/Reshape_grad/Shape',
     u'gradients/cross_entropy/Reshape_grad/Reshape',
     u'gradients/fc_layer_2/logits_output_grad/Shape',
     u'gradients/fc_layer_2/logits_output_grad/Shape_1',
     u'gradients/fc_layer_2/logits_output_grad/BroadcastGradientArgs',
     u'gradients/fc_layer_2/logits_output_grad/Sum',
     u'gradients/fc_layer_2/logits_output_grad/Reshape',
     u'gradients/fc_layer_2/logits_output_grad/Sum_1',
     u'gradients/fc_layer_2/logits_output_grad/Reshape_1',
     u'gradients/fc_layer_2/logits_output_grad/tuple/group_deps',
     u'gradients/fc_layer_2/logits_output_grad/tuple/control_dependency',
     u'gradients/fc_layer_2/logits_output_grad/tuple/control_dependency_1',
     u'gradients/fc_layer_2/MatMul_grad/MatMul',
     u'gradients/fc_layer_2/MatMul_grad/MatMul_1',
     u'gradients/fc_layer_2/MatMul_grad/tuple/group_deps',
     u'gradients/fc_layer_2/MatMul_grad/tuple/control_dependency',
     u'gradients/fc_layer_2/MatMul_grad/tuple/control_dependency_1',
     u'gradients/dropout/mul_grad/Shape',
     u'gradients/dropout/mul_grad/Shape_1',
     u'gradients/dropout/mul_grad/BroadcastGradientArgs',
     u'gradients/dropout/mul_grad/mul',
     u'gradients/dropout/mul_grad/Sum',
     u'gradients/dropout/mul_grad/Reshape',
     u'gradients/dropout/mul_grad/mul_1',
     u'gradients/dropout/mul_grad/Sum_1',
     u'gradients/dropout/mul_grad/Reshape_1',
     u'gradients/dropout/mul_grad/tuple/group_deps',
     u'gradients/dropout/mul_grad/tuple/control_dependency',
     u'gradients/dropout/mul_grad/tuple/control_dependency_1',
     u'gradients/dropout/div_grad/Shape',
     u'gradients/dropout/div_grad/Shape_1',
     u'gradients/dropout/div_grad/BroadcastGradientArgs',
     u'gradients/dropout/div_grad/RealDiv',
     u'gradients/dropout/div_grad/Sum',
     u'gradients/dropout/div_grad/Reshape',
     u'gradients/dropout/div_grad/Neg',
     u'gradients/dropout/div_grad/RealDiv_1',
     u'gradients/dropout/div_grad/RealDiv_2',
     u'gradients/dropout/div_grad/mul',
     u'gradients/dropout/div_grad/Sum_1',
     u'gradients/dropout/div_grad/Reshape_1',
     u'gradients/dropout/div_grad/tuple/group_deps',
     u'gradients/dropout/div_grad/tuple/control_dependency',
     u'gradients/dropout/div_grad/tuple/control_dependency_1',
     u'gradients/ReLU3_grad/ReluGrad',
     u'gradients/fc_layer_1/add_grad/Shape',
     u'gradients/fc_layer_1/add_grad/Shape_1',
     u'gradients/fc_layer_1/add_grad/BroadcastGradientArgs',
     u'gradients/fc_layer_1/add_grad/Sum',
     u'gradients/fc_layer_1/add_grad/Reshape',
     u'gradients/fc_layer_1/add_grad/Sum_1',
     u'gradients/fc_layer_1/add_grad/Reshape_1',
     u'gradients/fc_layer_1/add_grad/tuple/group_deps',
     u'gradients/fc_layer_1/add_grad/tuple/control_dependency',
     u'gradients/fc_layer_1/add_grad/tuple/control_dependency_1',
     u'gradients/fc_layer_1/MatMul_grad/MatMul',
     u'gradients/fc_layer_1/MatMul_grad/MatMul_1',
     u'gradients/fc_layer_1/MatMul_grad/tuple/group_deps',
     u'gradients/fc_layer_1/MatMul_grad/tuple/control_dependency',
     u'gradients/fc_layer_1/MatMul_grad/tuple/control_dependency_1',
     u'gradients/Reshape_grad/Shape',
     u'gradients/Reshape_grad/Reshape',
     u'gradients/max_pool_2x2_2_grad/MaxPoolGrad',
     u'gradients/ReLU2_grad/ReluGrad',
     u'gradients/conv_layer_2/add_grad/Shape',
     u'gradients/conv_layer_2/add_grad/Shape_1',
     u'gradients/conv_layer_2/add_grad/BroadcastGradientArgs',
     u'gradients/conv_layer_2/add_grad/Sum',
     u'gradients/conv_layer_2/add_grad/Reshape',
     u'gradients/conv_layer_2/add_grad/Sum_1',
     u'gradients/conv_layer_2/add_grad/Reshape_1',
     u'gradients/conv_layer_2/add_grad/tuple/group_deps',
     u'gradients/conv_layer_2/add_grad/tuple/control_dependency',
     u'gradients/conv_layer_2/add_grad/tuple/control_dependency_1',
     u'gradients/conv_layer_2/conv2d_2_grad/Shape',
     u'gradients/conv_layer_2/conv2d_2_grad/Conv2DBackpropInput',
     u'gradients/conv_layer_2/conv2d_2_grad/Shape_1',
     u'gradients/conv_layer_2/conv2d_2_grad/Conv2DBackpropFilter',
     u'gradients/conv_layer_2/conv2d_2_grad/tuple/group_deps',
     u'gradients/conv_layer_2/conv2d_2_grad/tuple/control_dependency',
     u'gradients/conv_layer_2/conv2d_2_grad/tuple/control_dependency_1',
     u'gradients/max_pool_2x2_1_grad/MaxPoolGrad',
     u'gradients/ReLu1_grad/ReluGrad',
     u'gradients/conv_layer_1/add_grad/Shape',
     u'gradients/conv_layer_1/add_grad/Shape_1',
     u'gradients/conv_layer_1/add_grad/BroadcastGradientArgs',
     u'gradients/conv_layer_1/add_grad/Sum',
     u'gradients/conv_layer_1/add_grad/Reshape',
     u'gradients/conv_layer_1/add_grad/Sum_1',
     u'gradients/conv_layer_1/add_grad/Reshape_1',
     u'gradients/conv_layer_1/add_grad/tuple/group_deps',
     u'gradients/conv_layer_1/add_grad/tuple/control_dependency',
     u'gradients/conv_layer_1/add_grad/tuple/control_dependency_1',
     u'gradients/conv_layer_1/conv2d_1_grad/Shape',
     u'gradients/conv_layer_1/conv2d_1_grad/Conv2DBackpropInput',
     u'gradients/conv_layer_1/conv2d_1_grad/Shape_1',
     u'gradients/conv_layer_1/conv2d_1_grad/Conv2DBackpropFilter',
     u'gradients/conv_layer_1/conv2d_1_grad/tuple/group_deps',
     u'gradients/conv_layer_1/conv2d_1_grad/tuple/control_dependency',
     u'gradients/conv_layer_1/conv2d_1_grad/tuple/control_dependency_1',
     u'beta1_power/initial_value',
     u'beta1_power',
     u'beta1_power/Assign',
     u'beta1_power/read',
     u'beta2_power/initial_value',
     u'beta2_power',
     u'beta2_power/Assign',
     u'beta2_power/read',
     u'conv_layer_1/W_conv_1/Adam/Initializer/zeros',
     u'conv_layer_1/W_conv_1/Adam',
     u'conv_layer_1/W_conv_1/Adam/Assign',
     u'conv_layer_1/W_conv_1/Adam/read',
     u'conv_layer_1/W_conv_1/Adam_1/Initializer/zeros',
     u'conv_layer_1/W_conv_1/Adam_1',
     u'conv_layer_1/W_conv_1/Adam_1/Assign',
     u'conv_layer_1/W_conv_1/Adam_1/read',
     u'conv_layer_1/b_conv_1/Adam/Initializer/zeros',
     u'conv_layer_1/b_conv_1/Adam',
     u'conv_layer_1/b_conv_1/Adam/Assign',
     u'conv_layer_1/b_conv_1/Adam/read',
     u'conv_layer_1/b_conv_1/Adam_1/Initializer/zeros',
     u'conv_layer_1/b_conv_1/Adam_1',
     u'conv_layer_1/b_conv_1/Adam_1/Assign',
     u'conv_layer_1/b_conv_1/Adam_1/read',
     u'conv_layer_2/W_conv_2/Adam/Initializer/zeros',
     u'conv_layer_2/W_conv_2/Adam',
     u'conv_layer_2/W_conv_2/Adam/Assign',
     u'conv_layer_2/W_conv_2/Adam/read',
     u'conv_layer_2/W_conv_2/Adam_1/Initializer/zeros',
     u'conv_layer_2/W_conv_2/Adam_1',
     u'conv_layer_2/W_conv_2/Adam_1/Assign',
     u'conv_layer_2/W_conv_2/Adam_1/read',
     u'conv_layer_2/b_conv_2/Adam/Initializer/zeros',
     u'conv_layer_2/b_conv_2/Adam',
     u'conv_layer_2/b_conv_2/Adam/Assign',
     u'conv_layer_2/b_conv_2/Adam/read',
     u'conv_layer_2/b_conv_2/Adam_1/Initializer/zeros',
     u'conv_layer_2/b_conv_2/Adam_1',
     u'conv_layer_2/b_conv_2/Adam_1/Assign',
     u'conv_layer_2/b_conv_2/Adam_1/read',
     u'fc_layer_1/W_fc_1/Adam/Initializer/zeros',
     u'fc_layer_1/W_fc_1/Adam',
     u'fc_layer_1/W_fc_1/Adam/Assign',
     u'fc_layer_1/W_fc_1/Adam/read',
     u'fc_layer_1/W_fc_1/Adam_1/Initializer/zeros',
     u'fc_layer_1/W_fc_1/Adam_1',
     u'fc_layer_1/W_fc_1/Adam_1/Assign',
     u'fc_layer_1/W_fc_1/Adam_1/read',
     u'fc_layer_1/b_fc_1/Adam/Initializer/zeros',
     u'fc_layer_1/b_fc_1/Adam',
     u'fc_layer_1/b_fc_1/Adam/Assign',
     u'fc_layer_1/b_fc_1/Adam/read',
     u'fc_layer_1/b_fc_1/Adam_1/Initializer/zeros',
     u'fc_layer_1/b_fc_1/Adam_1',
     u'fc_layer_1/b_fc_1/Adam_1/Assign',
     u'fc_layer_1/b_fc_1/Adam_1/read',
     u'fc_layer_2/W_fc_2/Adam/Initializer/zeros',
     u'fc_layer_2/W_fc_2/Adam',
     u'fc_layer_2/W_fc_2/Adam/Assign',
     u'fc_layer_2/W_fc_2/Adam/read',
     u'fc_layer_2/W_fc_2/Adam_1/Initializer/zeros',
     u'fc_layer_2/W_fc_2/Adam_1',
     u'fc_layer_2/W_fc_2/Adam_1/Assign',
     u'fc_layer_2/W_fc_2/Adam_1/read',
     u'fc_layer_2/b_fc_2/Adam/Initializer/zeros',
     u'fc_layer_2/b_fc_2/Adam',
     u'fc_layer_2/b_fc_2/Adam/Assign',
     u'fc_layer_2/b_fc_2/Adam/read',
     u'fc_layer_2/b_fc_2/Adam_1/Initializer/zeros',
     u'fc_layer_2/b_fc_2/Adam_1',
     u'fc_layer_2/b_fc_2/Adam_1/Assign',
     u'fc_layer_2/b_fc_2/Adam_1/read',
     u'Adam/learning_rate',
     u'Adam/beta1',
     u'Adam/beta2',
     u'Adam/epsilon',
     u'Adam/update_conv_layer_1/W_conv_1/ApplyAdam',
     u'Adam/update_conv_layer_1/b_conv_1/ApplyAdam',
     u'Adam/update_conv_layer_2/W_conv_2/ApplyAdam',
     u'Adam/update_conv_layer_2/b_conv_2/ApplyAdam',
     u'Adam/update_fc_layer_1/W_fc_1/ApplyAdam',
     u'Adam/update_fc_layer_1/b_fc_1/ApplyAdam',
     u'Adam/update_fc_layer_2/W_fc_2/ApplyAdam',
     u'Adam/update_fc_layer_2/b_fc_2/ApplyAdam',
     u'Adam/mul',
     u'Adam/Assign',
     u'Adam/mul_1',
     u'Adam/Assign_1',
     u'Adam',
     u'ArgMax/dimension',
     u'ArgMax',
     u'ArgMax_1/dimension',
     u'ArgMax_1',
     u'Equal',
     u'Cast',
     u'Const',
     u'Mean',
     u'init']




```python

```
