# User stories

## User persona: Medical Researcher
Might be a doctor, a neuroscientist, a biologist, a pharmaceutical researcher or
anyone interested in conducting research with medical data.
Might have a background in statistics/machine learning/data science, ranging
from basic to expert-level.

* A *medical researcher* wants to perform **basic federated analysis** on a set
  of *medical centers*<sup id="a1">[1](#f1)</sup>. This means running one
  *algorithm*<sup id="a2">[2](#f2)</sup> on a dataset which is scattered across
  multiple medical centers.
* A *medical researcher* wants to use a previously trained model to perform
  **prediction** on some new medical records.
* A *medical researcher* wants to perform **similarity search** across different
  medical centers. Given a small number (possibly one) of patients located in
  some medical center or whose medical record is inputted manually by the user,
  find other medical centers hosting *similar* cases. Similarity can have
  multiple definitions.
* A *medical researcher* wants to **cross-validate** her models using a
  variety of CV. The user wants to choose among multiple CV strategies.
    * *k-fold* CV Medical-center-agnostic.
    * *Cluster* CV where a cluster here is a medical center.
    * Leave Observation Out from each Cluster *(Wu and Zhang, 2002)*.
* A *medical researcher* wants to perform formalized **model selection**. After
  training multiple models on the same dataset, she wants to select the *best*
  one according to criteria usually related to the model's
  prediction/cross-validation performance.
* A *medical researcher* wants to be able to **pre-process data** using a variety
  of tools. 
    * In simple cases the pre-processing can be done **locally** in each
      medical center. *E.g.* creating interaction terms, *i.e.* creating new columns
      as products of existing ones.
    * In other cases the pre-processing itself needs to happen **globally**.
      *E.g.* centering or standatdizing data.
* A *medical researcher* wants to run **algorithm pipelines**. Typical flow: one
  algorithm run results in a model. This model is used to transform some data
  (*e.g.* via prediction). The transformed data is inputted to the next
  algorithm etc.
* A *medical researcher* wants to **save elements of her computations** for
  later use. These element can be models, data-views, transformed datasets etc.

## User persona: Algorithm Developer
A third party algorithm developer (as opposed to the developer of native
algorithms).

* An **algorithm developer** wants to supply the federation with her own
  algorithm, written in some machine learning friendly language (python, R,
  ...). Ideally:
    * The user is agnostic of the inner workings of the system, only being aware of
      the existence of many local and one global nodes.
    * The system enforces privacy restrictions independently of how the
      algorithm is written.

---
<b id="f1">1</b> A **medical center** is understood as being a member of the
federation. It stores and contributes data in the form of medical records as
well as some computing power. It is potentially geographically distinct from
other medical centers.[↩](#a1) 

<b id="f2">2</b> An **algorithm** is understood as being a **statistics/machine
learning** algorithm which typically excepts some data table as input and
outputs a **model**, *i.e.* a bunch of parameters. This model can be of interest
to the user on its own, or can be used in further computations.[↩](#a2) 
