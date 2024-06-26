<b><h2><center>Paired T-Test  </center></h1></b>

<b><h4> Some General Remarks </h4></b>
The general architecture of the MIP follows a Master/Worker paradigm where many Workers, operating in multiple medical centers, are coordinated by one Master. Only Workers are allowed access to the anonymized data in each medical center and the Master only sees aggregate data, derived from the full data and sent to him by the Workers.

As a consequence, every algorithm has to be refactored in a form that fits this model.

In general, this means two things.

1. On the one hand, isolating the parts of the algorithm that operate on the full data and implement them in procedures that run on Workers.
1. On the other hand, identifying the parts of the algorithm that need to see the aggregates from all Workers and implementing these parts in procedures that run on Master.

Our naming convention is that procedures run on Workers are given the adjective _local_ whereas those running on Master are called _global_.

<b><h4> Notation </h4></b>
Each local dataset *D<sup>(l)</sup>*, where *l*=1,...,*L*, is represented as a matrix of size *n* x *p*, where *L* is the number of medical centers, *n* is the number of points (patients) and *p* is the number of  attributes. The elements of the above matrix can either be continuous or discrete (categorical).

In each local dataset, the independent attributes are denoted as a matrix *X<sup>(l)</sup>* and the dependent variable is denoted as a vector *y<sup>(l)</sup>*. *x*<sub>(*ij*)</sub><sup>(*l*)</sup> is the value of the *i*<sup>(*th*)</sup> patient of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital, while *x*<sub>(*j*)</sub><sup>(*l*)</sup> denotes the vector of the *j*<sup>(*th*)</sup> attribute in the *l*<sup>(*th*)</sup> hospital. For categorical attributes,  we use the notation *C*<sub>m</sub> <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> { *C*<sub>1</sub>, *C*<sub>2</sub>, ..., *C*<sub>M</sub>} for their domain.

<b><h4> Algorithm Description </h4></b>
The Student’s paired samples t-test (sometimes called a dependent-samples t-test) is used to test the null hypothesis that the difference between pairs of measurements is equal to zero. A low p-value suggests that the null hypothesis is not true, and that the difference between the measurement pairs is not zero. In each local dataset, let *x<sub>j1</sub>* and *x<sub>j2</sub>* be the variables of interest.

![pseudo](images/paired_ttest_pseudocode.png)

<b><h4>Algorithm Implementation</b></h4>

[Paired T-test](../../exareme2/algorithms/exareme2/ttest_paired.py)
