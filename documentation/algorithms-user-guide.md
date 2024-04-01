# Federated Algorithms User Guide

- [Intro](#intro)
  - [System Overview](#system-overview)
- [Getting started](#getting-started)
  - [Federated algorithms overview](#federated-algorithms-overview)
  - [Writing a federated algorithm](#writing-a-federated-algorithm)
    - [Local computations](#local-computations)
    - [Global computations](#global-computations)
    - [Algorithm flow](#algorithm-flow)
  - [Porting to Exareme](#porting-to-exareme)
    - [Local and global steps as database UDFs](#local-and-global-steps-as-database-udfs)
    - [Algorithm flow in Exareme](#algorithm-flow-in-exareme)
    - [Data Loader](#data-loader)
    - [Algorithm Specifications](#algorithm-specifications)
    - [Running the algorithm](#running-the-algorithm)
- [Advanced topics](#advanced-topics)
  - [UDF generator](#udf-generator)
    - [API](#api)
    - [Multiple outputs](#multiple-outputs)
  - [Secure multi-party computation](#secure-multi-party-computation)
- [Best practices](#best-practices)
  - [Memory efficiency](#memory-efficiency)
  - [Time efficiency](#time-efficiency)
  - [Privacy](#privacy)
    - [Privacy threshold](#privacy-threshold)
    - [Share only aggregates](#share-only-aggregates)
- [Examples](#examples)
  - [Iterative algorithm](#iterative-algorithm)
  - [Class based model](#class-based-model)
  - [Complex data loader](#complex-data-loader)

# Intro

This guide will walk you through the process of writing federated algorithms
using Exareme. Exareme's focus is primarily on machine learning and statistical
algorithms, but the framework is so general as to allow any type of algorithm
to be implemented, provided that the input data is scattered across multiple
decentralized data sources.

## System Overview

Exareme consists of multiple services distributed across remote workers. There
are multiple local workers and a single global worker. Local workers host primary
data sources, while the global worker is responsible for receiving data from
local workers and perform global computations.

# Getting started

## Federated algorithms overview

In the highest level, a federated algorithm is composed of three ingredients.
Local computations, global computations and the algorithm flow. A local
computation takes places in each local worker and usually produces some aggregate
of the primary data found in the local worker. A global computation takes place
in a special worker called the global worker, and usually consolidates the local
aggregates into a global aggregate. Finally, the algorithm flow is responsible
for coordinating these computations and the data exchange between workers.

## Writing a federated algorithm

In order to write a federated algorithm we need to define local and global
computations and write an algorithm flow that ties the computations together.

Let's break down the steps one by one. We'll begin by writing a simple
algorithm for computing the mean of a single variable. This algorithm will be
written as a simple python script, running on a single machine. The purpose of
this exercise is to illustrate how an algorithm is decomposed into local and
global steps, and how the flow coordinates these steps. Later, we'll add the
necessary ingredients in order to be able to run the algorithm in an actual
federated environment, using Exareme.

Since we will run the algorithm on a single machine we will represent the
federated data as a list of dataframes `data: list[pandas.DataFrame]`.

#### Local computations

A local computation is python function taking local data and returning a
dictionary of local aggregates. The example below demonstrates a local
computation function that calculates the sum and count of some variable.

```python
def mean_local(local_data):
    # Compute sum and count
    sx = local_data.sum()
    n = len(local_data)

    # Pack results into single dictionary which will
    # be transferred to global worker
    results = {"sx": sx, "n": n}
    return results
```

The result packs the two aggregates, `sx` and `n` into a dictionary. A separate
instance of this function will run on each local worker, so `sx` and `n` are
different for each worker and reflect each worker's local data.

#### Global computations

A global computation is also a python function, and it usually takes the output
of a local computation as an input. The local computations, coming from
multiple workers, produce a list of results, from which a global aggregate is
computed. We can then perform further computations, as in the current example
where the mean is computed by dividing the global `sx` with the global `n`.

```python
def mean_global(local_results):
    # Sum aggregates from all workers
    sx = sum(res["sx"] for res in local_results)
    n = sum(res["n"] for res in local_results)

    # Compute global mean
    mean = sx / n

    return mean
```

#### Algorithm flow

The algorithm flow coordinates local and global computations, as well as the
exchange of data between workers. We can write the algorithm flow as a python function
`run` that calls `mean_local` on all local workers and then calls `mean_global` on the
global worker, passing the results of the local computations.

```python
def run(all_data):
    local_results = [mean_local(local_data) for local_data in all_data]
    mean = mean_global(local_results)
    return mean
```

We can then run the algorithm flow like this:

```python
import pandas as pd

x1 = pd.DataFrame({"var": [1, 2, 3]})
x2 = pd.DataFrame({"var": [4, 5, 6]})
all_data = [x1, x2]
print(run(all_data))
```

## Porting to Exareme

Let's now port the above algorithm to Exareme so that it can be run in a real
federated environment. We will still write functions to represent local and
global computations, but we need to take a few extra steps to help Exareme run
in the federated environment.

### Local and global steps as database UDFs

The first this we need to do is to inform Exareme about which functions should
be treated as local and global computations. This is done by the `udf`
decorator, imported from the `udfgen` module. The name UDF stands for **user
defined function** and comes from the fact that the decorated functions will
run as database UDFs.

More importantly, we also need to inform Exareme about the *types* of variables
involved in the local and global computations. Python is a dynamic language
where type annotations are optional. On the other hand, code written for
Exareme will run in an environment which is not just a single Python
interpreter. The various local and global steps will run in separate
interpreters, each embedded in the corresponding relational database. These
computations will need to communicate with the database in order to read from
and write data to it. Moreover, the outputs of these local and global
computations can have different fates. Some will be sent across the network to
other workers, while others will be stored in the same worker for later processing.
Having variables with dynamic types would make the communication with the
database and the communication between workers very difficult to implement
efficiently. To overcome these difficulties, the `udfgen` module defines a
number of *types* for the input and output variables of local and global
computations.

Let's rewrite the local/global functions of the previous examples as Exareme
UDFs. First the local UDF.

```python
from exareme2.algorithms.exareme2.udfgen import udf, relation, transfer


@udf(local_data=relation(), return_type=transfer())
def mean_local(local_data):
    # Compute two aggregates, sx and n_obs
    sx = local_data.sum()
    n = len(local_data)

    # Pack results into single dictionary which will
    # be transferred to global worker
    results = {"sx": sx, "n": n}
    return results
```

The actual function is exactly the same as before, the difference lies in the
`udf` decorator. `local_data` is declared to be of type `relation`. This means
that the variable will be a relational table, implemented in python as a pandas
dataframe. The output is of type `transfer`. This means that we intend to
transfer the output to another worker. In our python implementation this is a
plain dictionary but it will be converted to a JSON object in order to be
transferred. This means that the contents of the dictionary should be JSON
serializable.

Now, let's write the global UDF.

```python
from exareme2.algorithms.exareme2.udfgen import udf, transfer, merge_transfer


@udf(local_results=merge_transfer(), return_type=transfer())
def mean_global(local_results):
    # Sum aggregates from all workers
    sx = sum(res["sx"] for res in local_results)
    n = sum(res["n"] for res in local_results)

    # Compute global mean
    mean = sx / n

    # Pack result into dictionary
    result = {"mean": mean}
    return result
```

The type of `local_results` is `merge_transfer`. This means that the
`local_results` will be a list of dictionaries corresponding to one
`mean_local` output per worker. The return type is now again of type `transfer`
since, unlike in the single-machine example, we now need to transfer the global
result to the algorithm flow which might run in a different machine.

### Algorithm flow in Exareme

Finally, lets write the algorithm flow. This will be quite quite different from
the single-machine case. The flow is encapsulated as a python object exposing a
`run` method. This object is instantiated by the Exareme algorithm execution
engine, which is the mechanism for executing federated algorithms and takes
care of relaying work to the workers and routing the transfer of data between
workers. As algorithm writers, we need to inform the algorithm execution
engine about UDF execution order and data transfer, and this is done through
the algorithm execution interface. To have access to this interface we have to
inherit from the `Algorithm` base class.

```python
from exareme2.algorithms.exareme2.algorithm import Algorithm


class MyAlgorithm(Algorithm, algname="my_algorithm"):
    def run(self, data, metadata):
        local_results = self.engine.run_udf_on_local_workers(
            func=mean_local,
            keyword_args={"local_data": data},
            share_to_global=True,
        )
        result = self.engine.run_udf_on_global_worker(
            func=mean_global,
            keyword_args={"local_results": local_results},
        )
        return result
```

The attribute `engine`, inherited from `Algorithm`, has two methods for calling
UDFs. `run_udf_on_local_workers` runs a particular UDF **on all local workers**,
each with the corresponding local data. `run_udf_on_global_worker` runs a UDF
**on the global worker**.

Since we want the local results to be transferred to the global worker
for further computations, we have to pass the `share_to_global=True` argument
to the first method.

### Data Loader

One issue that did not come up in the single machine version is data loading.
In the single machine version this is a trivial operation. However, in the
federated case, the actual data content has some essential impact on the
algorithm orchestration. For a particular data choice by the user, all workers
having no data, or having data below some [**privacy
threshold**](#privacy-threshold) will not participate in the run. This is
something that Exareme needs to know before the start of the algorithm
execution. This is achieved by defining a separate class, extending
`AlgorithmDataLoader`, where the algorithm writer implements the logic
according to which the data are loaded from the database into python
dataframes.

The main method to implement is `get_variable_groups` which returns a
`list[list[str]]`. The inner list represents a list of column names,
while the outer one a list of dataframes. The user requested column names
can be found in `self._variables`.

In our case we need a very simple data loader for a single dataframe with a
single column, as requested by the user (see Examples for more advanced uses).

```python
from exareme2.algorithms.exareme2.algorithm import AlgorithmDataLoader


class MyDataLoader(AlgorithmDataLoader, algname="mean"):
    def get_variable_groups(self):
        return [self._variables.x]
```

### Algorithm Specifications

Finally, we need to define a specification for each algorithm. This contains
information about the public API of each algorithm, such as the number and
types of variables, whether they are required or optional, the names and types
of additional parameters etc.

The full description can be found in
[`exareme2.algorithms.specifications.AlgorithmSpecification`](https://github.com/madgik/exareme2/blob/algo-user-guide/exareme2/algorithms/specifications.py).
The algorithm writer needs to provide a JSON file, with the same name and
location as the file where the algorithm is defined. E.g. for an algorithm
defined in `dir1/dir2/my_algorithm.py` we also create
`dir1/dir2/my_algorithm.json`. The contents of the file are a JSON object with
the exact same structure as
`exareme2.algorithms.specifications.AlgorithmSpecification`.

In our example the specs are

```json
{
    "name": "mean",
    "desc": "Computes the mean of a single variable.",
    "label": "Mean",
    "enabled": true,
    "inputdata": {
        "x": {
            "label": "Variable",
            "desc": "A unique numerical variable.",
            "types": [ "real" ],
            "stattypes": [ "numerical" ],
            "notblank": true,
            "multiple": false
        }
    }
}
```

### Running the algorithm

Once all building blocks are in place, and our system is deployed, we can run
the algorithm either by performing a POST request to Exareme, or by using
[`run_algorithm`](https://github.com/madgik/exareme2/blob/algo-user-guide/run_algorithm) from the
command line.

# Advanced topics

The previous example is enough to get you started, but Exareme offers a few
more features, giving you the necessary tools to write more complex algorithms.
Let's explore some of these tools in this section.

## UDF generator

The [UDF generator module](https://github.com/madgik/exareme2/tree/algo-user-guide/exareme2/udfgen)
is responsible for translating the `udf` decorated python functions into actual
UDFs which run in the database. This translation has a few subtle points,
mostly related to the conflict between the dynamically typed Python on one
hand, and the statically typed SQL on the other. The `udfgen` module offers a
few types to be used as input/output types for UDFs. These encode information
about how to read or write a python object into the database.

### API

For a detailed explanation of the various types see the
[module's docstring](https://github.com/madgik/exareme2/blob/algo-user-guide/exareme2/udfgen/py_udfgenerator.py).
Here we present a few important ones.

##### `relation(schema=None)`

The type `relation` is used for relational tables. In the database these are
plain tables whereas in Python they are encoded as pandas dataframes. The
table's schema can be declared by passing the `schema` arg to the constructor,
e.g. `relation(schema=[('var1', int), ('var2', float)])`. If `schema` is
`None`, the schema is generic, i.e. it will work with any schema passed at
runtime.

##### `tensor(dtype, ndims)`

A tensor is an n-dimensional array. In Python these are encoded as `numpy`
arrays. Tensors are fundamentally different from relational tables in that
their types are homogeneous and the order of their rows matter. Tensors are
used when the algorithmic context is linear algebra, rather than relational
algebra. `dtype` is the tensor's datatype, and can be of type `type` or
`exareme2.datatypes.DType`. `ndims` is an `int` and defines the tensor's
dimensions. Another benefit of tensors is that their data are stored in a
contiguous block of memory (unlike `relations` where individual columns are
contiguous) which result in better efficiency when used within frameworks like
`numpy`, which makes heavy use of the vectorization capabilities of the CPU.

##### `literal()`

Literals are used to pass small, often scalar, values to UDFs. Examples are
single a `int`, `float` or `str`, a small `list` or `dict`. In general, values
for which it wouldn't make much sense to encode as tables in the database.
These are not passed to the UDF as inputs. They are instead printed literally
to the UDF's code, hence the name.

##### `transfer()`

Transfer objects are used to send data to and from local/global workers. In
Python they are plain dictionaries, but they are transformed to JSON for the
data transfer, so all values in the `dict` must be JSON serializable, and all
keys must be strings. `transfer` does not encrypt data for
[SMPC](#secure-multi-party-computation) and thus should be used for
non-sensible data and for sending data from the global worker to the local workers.

##### `secure_transfer(sum_op=False, min_op=False, max_op=False)`

Type used for sending data thought the SMPC cluster. See
[SMPC](#secure-multi-party-computation) for more details.

##### `state()`

State objects are used to store data in the same worker where they are produced,
for later consumption. Like `transfer`/`secure_transfer`, they are Python
dictionaries but they are serialized as binary objects using `pickle`.

### Multiple outputs

A UDF can also have multiple outputs of different kinds. The typical use-case
is when we want to store part of the output locally for later use in the same
worker, and we want to transfer the other part of the output to another worker.

```python
from exareme2.algorithms.exareme2.udfgen import udf, state, transfer


@udf(input=relation(), return_type=[state(), transfer()])
def two_outputs(input):
    ...  # compute stuff
    output_state = {}  # output_state is a dict where we store variables for later use
    ...  # add stuff to output_state
    output_transfer = {}  # output_transfer is a dict with variables we want to transfer
    ...  # add stuff to output_transfer
    return output_state, output_transfer  # multiple return statement
```

Note that this time we declared a list of outputs in the `udf` decorator. Then,
we simply use the Python multiple return statement and Exareme takes care of
the rest. In the database, the first element of the return statement is
returned as a table (the usual way of returning things from a UDF), while the
remaining elements are returned through the loopback query mechanism, which is
a mechanism for executing database queries from within the code of the UDF. It
is therefore advised to place the object with the largest memory footprint
first in the list of returned objects.

## Secure multi-party computation

Secure multi-party computation is a cryptographic technique used when multiple
parties want to jointly perform a computation on their data, usually some form
of aggregation, but wish for their individual data to remain private. For more
details see
[Wikipedia](https://en.wikipedia.org/wiki/Secure_multi-party_computation). In
Exareme, SMPC is used to compute global aggregates. When a global aggregate
must be computed, all participating local workers first compute the local
aggregates. These are then fed to the SMPC cluster in an encrypted form. The
SMPC cluster performs the global aggregation and sends the result back to
Exareme, where it is passed as an input to the global UDF.

To implement a SMPC computation we need to have a local UDF with a
`secure_transfer` output.

```python
from exareme2.algorithms.exareme2.udfgen import udf, relation, secure_transfer


@udf(local_data=relation(), return_type=secure_transfer(sum_op=True))
def mean_local(local_data):
    sx = local_data.sum()
    n = len(local_data)

    results = {"sx": {"data": sx, "operation": "sum", "type": float},
               "n": {"data": n, "operation": "sum", "type": int}}
    return results
```

First we have to activate one or more aggregation operations. Here we activate
summation, passing `sum_op=True`. Then we have to pass some more information to
the output dict. Namely, the data, the operation used in the aggregation, and
the datatype. Values for the `"data"` key can be scalars or nested lists.

The global UDF then needs to declare its input using `transfer`.

```python
@udf(local_results=transfer(), return_type=transfer())
def mean_global(local_results):
    sx = local_results['sx']
    n = local_results['n']

    mean = sx / n
    result = {"mean": mean}
    return result
```

Note that we don't need to perform the actual summation, as we did previously,
as it is now performed by the SMPC cluster.

# Best practices

## Memory efficiency

The whole point of translating Python functions into database UDFs (see [UDF
generator](#udf-generator)) is to avoid unnecessary data copying, as the
database can transfer data to UDFs with zero cost. If we are not careful when
writing `udf` functions, we could end up performing numerous unnecessary copies
of the original data, effectively canceling the zero-cost transfer.

For example, say we want to compute the product of three matrices, `A`, `B` and
`C`, and then sum the elements of the final resulting matrix. The result is a
single float and we ought to be able to allocate just a single float. If we are
not careful and write

```python
result = (A @ B @ C).sum()
```

Python will allocate a new matrix for the result of `A @ B`, then another one for `A @ B @ C`
and then it will sum the elements of this last matrix to obtain the final result.

To overcome this we can use `numpy.einsum` like this

```python
result = numpy.einsum('ij,jk,kl->', A, B, C)
```

This will only allocate a single float!

There are multiple tricks like this one that we can use to reduce the memory
footprint of our UDFs. You can find a few in this
[commit](https://github.com/madgik/exareme2/commit/5d15847898930ac44fcf3be3669b4e74f427d5b5)
together with a short summary of the main ideas.

## Time efficiency

When writing a federated algorithm, we usually decompose an existing algorithm
into local and global computations. This decomposition is not unique in
general, thus the same algorithm can be written in more than one ways. However,
different decompositions might lead to important differences in execution time.
This is related to the overall number of local and global steps. Let's call a
sequence of one local and one global step, a **federation round**.. Each
federation round, together with the required data interchange between workers,
takes a non-negligible amount of time. It is therefore desirable to find a
decomposition that minimizes the number of federation rounds.

Consider, for example, the computation of the [total sum of
squares](https://en.wikipedia.org/wiki/Total_sum_of_squares). This quantity is
needed in many algorithms, such as linear regression or ANOVA. The total sum of
squares (TSS) is given by

```math
\text{TSS} = \sum_i^N (y_i - \hat{y})^2
```

We might think that we have to compute $\\hat{y}$ in a single round, then share the
result with the local workers, and finally compute the TSS in a second round.
But in fact, the whole computation can be done in a single round. We first develop
the square of the difference.

```math
\text{TSS} = \sum_i^N y_i^2 - 2 \hat{y} \sum_i^N y_i + N \hat{y}^2
```

It follows that we must compute the sum, the sum of squares and $N$, in the
local step. Then, in the global step, we can compute $\\hat{y}$ and the above
expression for the TSS. We managed to compute the result in a single federation
round, instead of two.

## Privacy

Privacy is a very subtle subject. In the context of federated analytics it
means to protect individual datapoints residing in local workers, as these might
correspond to a person's sensitive personal data. Exareme allows the user to
learn something about a large group of individuals through statistics, but
tries to prevent any individual's information to leak.

In order to achieve this we follow a few guidelines to ensure maximum
protection of sensible data.

#### Privacy threshold

First, there is a threshold of datapoints that a worker needs in order to
participate in a computation. No worker with datapoints below this threshold is
allowed to participate. This threshold is system wide (its value is usually
around 10) and the algorithm writer has no control over it.

#### Share only aggregates

The most important guideline you should follow, as an algorithm writer, is to
allow local workers to share only aggregates of the sensitive data they hold, and
never the actual data! Every variable in the database contains multiple values,
one for each individual. When some variable is involved in a computation we
must make sure that we only share the result of some function that aggregates
these multiple values into a single value.

In (informal) mathematical language, we can say that, given a number of datapoints $N$,
we want functions whose output size is $O(1)$ and not $O(N)$. For example
`sum`, when applied to a continuous variable, is $\\texttt{sum} : \\mathbb{R}^N \\rightarrow \\mathbb{R}$
and `count`, when applied to a categorical variable with categories $C$, is
$\\texttt{count} : C^N \\rightarrow \\mathbb{N}$. Both these functions have codomains that are
independent of $N$, so they are good examples of aggregate functions. Other examples are `min` and `max`.

These functions are not limited to plain sums of counts, however. Let's say,
for example, that we have a $N \\times p$ design matrix $\\mathbf{X}$, where
$N$ is the number of rows and $p$ is the number of variables. It is often the
case in statistics that we need to compute the quantity $\\mathbf{X}^\\intercal \\cdot \\mathbf{X}$.
Although not scalar, this quantity complies with our definition of aggregate,
since all elements of the resulting matrix are themselves aggregates over $N$,
hence the result size is $O(p^2)$ and doesn't depend on $N$.

# Examples

## Iterative algorithm

In the previous sections we presented a very simple algorithm for computing the
mean of some variable. This algorithm requires a single local and a single
global step. More complex workflows are possible, such as an
iterative workflow. Below is an example of how to structure code to achieve
this. For a real world example you should see the `fit` method in the [logistic regression
algorithm](https://github.com/madgik/exareme2/blob/algo-user-guide/exareme2/algorithms/logistic_regression.py).

```python
from exareme2.algorithms.exareme2.algorithm import Algorithm
from exareme2.algorithms.exareme2.helpers import get_transfer_data


class MyAlgorithm(Algorithm, algname="iterative"):
    def run(self, data, metadata):
        val = 0
        while True:
            local_results = self.engine.run_udf_on_local_workers(
                local_udf,
                keyword_args={"val": val},
                share_to_global=True,
            )
            result = self.engine.run_udf_on_global_node(
                global_udf,
                keyword_args={"local_transfers": local_results}
            )
            data = get_transfer_data(result)
            val = data["val"]
            criterion = data["criterion"]

            if criterion:
                break

        return val
```

Here we initialize `val` to 0 and start the iteration. In each step `val` is
passed to the local UDF. The local result is passed to the global UDF which
computes the new value for `val` and `criterion`. The value `criterion` decides
when the iteration stops.

## Class based model

When the algorithm logic becomes too complex, we might want to abstract some parts into separate
classes. This is possible and advised. For example, when the algorithm makes use of a supervised learning
model, e.g. [linear regression](https://github.com/madgik/exareme2/blob/algo-user-guide/exareme2/algorithms/linear_regression.py),
we can create a separate class for the model and call it from within the algorithm class.
Typically, a model implements two methods, `fit` and `predict`. The first does the learning by fitting the
model to the data, and the second does prediction on new data. Both methods could be used in a cross-validation
algorithm, see for example [here](https://github.com/madgik/exareme2/blob/algo-user-guide/exareme2/algorithms/linear_regression_cv.py).

```python
from exareme2.algorithms.exareme2.algorithm import Algorithm


class MyModel:
    def __init__(self, engine):
        self.engine = engine

    def fit(self, data):
        # Complex computation calling local and global UDFs though self.engine
        ...

    def predict(self, new_data):
        ...


class MyAlgorithm(Algorithm, algname="complex_algorithm"):
    def run(self, data, metadata):
        model = MyModel(self.engine)  # need to pass self.engine
        model.fit(data)

        new_data = ...
        predictions = model.predict(new_data)
        ...
```

## Complex data loader

In some cases we need to implement some complex logic during data loading. For example
we might need to consolidate all variables in X and Y into a single table, and add some
extra variable which will always appear in the table, even when not selected by the user.
We might also need, for example, to keep the NA values in the table.

This kind of complex logic can be implemented in the `AlgorithmDataLoader`
class, as in the example below.

```python
class DescriptiveStatisticsDataLoader(AlgorithmDataLoader, algname='my_algorithm'):
    def get_variable_groups(self):
        xvars = self._variables.x
        yvars = self._variables.y

        all_vars = xvars + yvars           # consolidate to single table
        all_vars.append('extra_variable')  # append extra variable

        return [all_vars]

    def get_dropna(self) -> bool:
        return False                       # do not drop NA from table
```
