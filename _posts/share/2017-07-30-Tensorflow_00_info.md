---
layout: post
title:  ""
date:   2017-07-30
categories: share
share: false
---


# Some information and resources
- IPython (interactive) and Jupyter(notebok for noting and sharing)
    - Good thing about Jupyter: integrate code and document(markdown/html/latex). 
    
    Ex. $softmax(x_i) = \frac{\exp^{x_i}}{\sum_{k=1}^K \exp^{x_k}}$
    - Jupyter shortcut keys: https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/


- Tensorflow and Virtual environment:  https://www.tensorflow.org/install/install_mac


- Using Jupyter under virtualenv: http://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs
    - Activate your virtual environment (on Mac)
    
      ```bash
      source <virtual_env_path>/bin/activate
      ```
      
      Note that you should see **(your_virtualenv_name)** before the prompt when the virtual environment is successfully activated. Here we use **(virtualenv)** as an example.
      
    - Install ipython kernel to the virtualenv
    
      ```bash
      (virtualenv) pip install ipykernel  # pip3 install ipykernel for python3
      ``` 
      
    - Create your kernel
    
      ```bash
      (virtualenv) python -m ipykernel install --user --name=<your_kernel_name>
      ```
      
    - Lauch Jupyter and then you can see the newly added kernel
    
      ```bash
      (virtualenv) jupyter notebook
      ```
      
- Tensorflow tutorial
    - Official tutorial: https://www.tensorflow.org/get_started/
    - Vedios about tensorflow (Chinese): https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/
    
    
- Tensorflow validation (hello world)


```python
import tensorflow as tf
hello = tf.constant('Hello, Tensorflow')
sess = tf.Session()
print(sess.run(hello))
```

    Hello, Tensorflow


- slim: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
- pre-trained model https://github.com/tensorflow/models/tree/master/slim#Pretrained
- GAN https://www.youtube.com/watch?v=0CKeqXl5IY0
