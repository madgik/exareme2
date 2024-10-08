<b><h2><center>Naive Bayes Classifier</center></h1></b>

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
In Naive Bayes algorithm the attributes of *X* can be both categorical and continuous, while the *y* is always categorical. Once we have the likelihood terms from the training procedure we can compute the maximum a posteriori probability for the class of a new query datapoint *q* with the following procedure:

![pseudo](images/nb_train_pseudocode.png)

![pseudo](images/nb_predict_pseudocode.png)

<b><h4>Algorithm Implementation</b></h4>

[Categorical Naive Bayes with Cross - Validation](../../exareme2/algorithms/exareme2/naive_bayes_categorical_cv.py)

[Gaussian Naive Bayes with Cross - Validation](../../exareme2/algorithms/exareme2/naive_bayes_gaussian_cv.py)
