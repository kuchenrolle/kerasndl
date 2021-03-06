# kerasndl

A keras implementation of the Naive Discriminative Learner network (NDL) to utilize GPU resources in psycholinguistic modelling.


## Getting Started

First you will need to clone the github directory, at which point you can start using it from inside python.

```shell
user@host:~ git clone https://github.com/kuchenrolle/kerasndl.git
user@host:~ cd kerasndl/src
user@host:~ python
```


## Main Functionality

Let's first process an example corpus (one sentence per line) into an event file, which is done with the **Preprocessor** class:

```python
>>> import numpy as np
>>> from kerasndl.preprocessing import Preprocessor

>>> path_to_corpus = "my_corpus.txt"
>>> path_to_event_file = "my_corpus.events"
>>> preprocessor = Preprocessor()
>>> preprocessor.process_file(path_to_corpus, path_to_event_file)
```

The events should now be saved in my_corpus.events and can be used with kerasndl's main class, the **Learner**, which handles the interfaces to the NDL network and the training events file, such that training and querying can go smoothly. Let's train the learner on the first 10 events we have created and extract the weights between some cues and outcomes:

```python
>>> from kerasndl.learner import Learner

>>> learner = Learner(path_to_event_file)
>>> learner.learn(10)

>>> cues = ["so","om","me"] # default cue size is letter uni- and bigrams
>>> outcomes = ["and","some","test","outcomes"]

>>> weights = learner.get_weights(cues = cues, outcomes = outcomes, named = True)
>>> weights
    and  outcomes      some      test
me  0.0       0.0  0.001844  0.001234
om  0.0       0.0  0.001844  0.001234
so  0.0       0.0  0.001844  0.001234
```

Setting named to True will result in the weights being return as a pandas data frame with named rows (cues) and columns (outcomes), otherwise they will be returned as a plain two-dimensional numpy array. Let's continue learning until all events have been processed, save the weights and get the status of the learner.

```python
>>> learner.learn() # not providing the number of events learns all remaining
>>> weights = learner.get_weights(cues = cues, outcomes = outcomes)
>>> np.save("my_weights.npy", weights)
>>> print(learner.info)
```

Alternatively, save the weights as a feather data frame, which allows to port them to R. Feather does not understand row names, though, so these are stored as the last column called "index" and must accordingly be added back if working with pandas.

```python
>>> learner.save_as_feather("my_weights.feather", cues, outcomes)
```

Finally, plot the weights (weights must now be a named pandas data frame) as a bipartite graph with connections color coding the weights, display and save it.

```python
>>> from kerasndl.visualize import plot_graph
>>> # import pandas as pd
>>> # weights = pd.read_feather("my_weights.feather")
>>> # weights.index = weights["index"] # set index
>>> # weights = weights.drop("index", 1) # remove column "index"

>>> plot_graph(weights, output_file = "my_plot.pdf")
```

This can sometimes lead to an ImportError, which appears to stem from some internal issues between tensorflow, spacy and matplotlib - if this happens, save the weights with feather and open in another session.
(Uncomment line two and three in that case.)
It can be avoided by importing plot_graph before doing any other imports.

### Performance

The highly-optimized reference implementation of NDL is that from [pyndl](https://pypi.python.org/pypi/pyndl/0.3.0), so kerasndl's perfomance should be plotted against that of pyndl. When trained on 20000 three-word phrases randomly drawn from the English SUBTLEX corpus to predict words from words (3800 cues and outcomes in total), results were identical, but the latencies differed dramatically:

**kerasndl (CPU):           822s**

**kerasndl (GPU):           240s**

**pyndl (Pure Python)       139s**

**pyndl (C, memory mapping) 2s**

When using the GPU, kerasndl is comparable in speed to that of the pyndl version, while it is much slower than the reference implementation when run on the CPU. But all of these are outperformed by pyndl's implementation in C, which gains its dramatic performance increase from mapping cues and outcomes in a preprocessing step to memory addresses, eliminating all look-up costs.

Even though kerasndl was developed to allow it to be easily modifiable using the tools that keras offers (e.g. different activation functions, mini-batch training, attention mechanisms) to investigate such changes affect NDL's performance, it was also intended to recruit resources (GPU) that are unavailable to pyndl, which the speed comparison suggests will not make much of a difference.

It is worth pointing out, though, that this is a proof-of-concept and it is anticipated that switching to a lower-level interface will eliminate a large portion of the overhead and allow kerasndl to break even with the performance of the pure python implementation. More importantly, relative to CPUs, GPUs have seen larger improvements in performance, which will further reduce the performance gap.

But the biggest reason for the difference in speed is the nature of the modelled problem. Events for NDL are typically very sparse, such that only few of a large number of cues are learnt to only few of a large number of outcomes on each event. The reference implementation updates each connection individually, but keras will calculate updates for all weights at once, which will produce a huge amount of overhead, as most of those updates will not be used. The way to overcome this is by implementing using CUDA instructions direcly, at which point using GPUs will likely become very efficient.

### Prerequisites

kerasndl has several dependencies that are not part of the standard distribution. They are **Numpy** (numerical computing), **Pandas** (data frame interface to Numpy), **Tensorflow** (High-level neural network library), **Keras** (even higher-level nn libary, convenient interface for tensorflow) and **Spacy** (NTLK's high-performing cousin). On Unix systems, you should be able to install all of these using pip:

```shell
user@host:~ pip install pandas # will install numpy automatically
user@host:~ pip install keras # will install its backend tensorflow automatically
user@host:~ pip install spacy
```

kerasndl also features some basic plotting capabilities, the usage of which requires an otherwise optional dependency, **networkx**. Install analogously:

```shell
user@host:~ pip install networkx
```

A highly-recommended alternative, especially for Windows users, is to use a conda distribution that should come shipped with pandas and add the remaining dependencies in a virtual environment like so:

```shell
user@host:~ [source] activate <environment>
(environment)user@host:~ conda install spacy
(environment)user@host:~ conda install keras
(environment)user@host:~ conda install networkx
```

The steps above will install the CPU version of keras and tensorflow. Enabling GPU support is a bit more involved and requires to set up interfaces to the graphic card, CUDA and CuDNN, a good tutorial on how to do that can be found [here](https://medium.com/@acrosson/installing-nvidia-cuda-cudnn-tensorflow-and-keras-69bbf33dce8a)

Once CUDA and CuDNN are installed, conda makes it easy to install the GPU version:

```shell
user@host:~ [source] activate <environment>
(environment)user@host:~ conda install keras-gpu
```


If you encounter any problems installing the dependencies, please consult the installation instructions for [Pandas](http://pandas.pydata.org/pandas-docs/stable/install.html), [Spacy](https://spacy.io/docs/usage/) and [keras](https://keras.io/).


## Author

**Christian Adam**


## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details