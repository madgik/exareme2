## Specification for privacy tools for the MIP

### Setting

The MIP is an tool for data analysis on the combined databases of a federation of medical centers across Europe.
Structurally it is composed of multiple local nodes, one for each medical center, and one global node which coordinates
the computation.

Each local node communicates solely with the global node and is allowed to share only aggregate information. The global
node acquires the local aggregates and merges them into a global aggregate. For simple queries (usually statistical
tests and processes) there is only one, or a small fixed number or local/global iterations. More complex queries (mostly
machine learning model training) usually require an unbounded iteration.

The merging stage in the global node depends on the particular query but we can formulate some general laws:

1. **Associativity**

```
(a + b) + c = a + (b + c)
```

It should not matter if the global node folds the local aggregates left-to-right or right-to-left.

2. **Commutativity**

```
a + b = b + a
```

We cannot guarantee the order by which the local nodes will broadcast their results, hence the order should be
irrelevant.

3. **Identity element**

```
a + e = e + a = a
```

An identity is useful when one local node cannot produce results if, for example, it doesn't have the data asked by the
query or if it is left out by the computation for some reason (e.g. in some cross-validation settings).

The algebraic structure described by those laws is that of a **Commutative Monoid**. Therefore, in the general case, the
local aggregates must be elements of some commutative monoid and the merging operation its corresponding law. In
practice, however, the majority of cases corresponds to simple summation of numbers. In some cases (e.g. to compute some
global descriptive statistics) we use the min/max operation over numbers. It can also happen that the local aggregates
are sets and the monoid operation is set union.

### Privacy tools

In order to provide privacy guaranties to the members of the federation, we wish to introduce a privacy mechanism in the
MIP. This mechanism will be composed of two layers. The first layer uses Secure Multi-Party Computation
(SMPC) to compute the global aggregate result. Its purpose is to minimize the leakage of information from the local
databases by making sure that local aggregates are never known to the global node. The second layer of privacy will
introduce some random noise to the final result as described by the Differential Privacy theory. The purpose of this
layer is to provide protection for individual medical records by providing guaranties against the use of differential
attacks on the federation.

Structurally, the first mechanism (SMPC) will be implemented as a component mediating the communication from the local
nodes to the global. This component should receive the local aggregates from the local nodes and output the merged
result (as described by the monoid laws above) to the global node.

The second mechanism is much simpler and can be a component that adds DP noise to the final result. In that simple case
this component sits between the global node and the User Interface. Arguably a better solution would be to add DP noise
already in the local aggregates so that if the global node is compromised DP guaranties still hold. In that case it
would be implemented as a component between the local node and the SMPC layer.
