## Federated Averaging

#### Some General Remarks

The general architecture of the MIP follows a Master/Worker paradigm where many Workers
, operating in multiple medical centers, are coordinated by one Master. Only Workers
are allowed access to the anonymized data in each medical center and the Master only
sees aggregate data, derived from the full data and sent to him by the Workers.

Our naming convention is that procedures run on Workers are given the adjective _local_
whereas those running on Master are called _global_.

In this premise, we implement federated averaging by building the models locally, using 
state-of-the-art Python libraries, such as scikit learn and then averaging the parameters
on the global node.

#### Algorithm Description

This algorithm aggregates the parameters of the local models and returns their average.

<b><h4>Algorithm Implementation</b></h4>

[FedAvg](../../exareme2/algorithms/in_database/fedaverage.py)
