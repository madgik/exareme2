# Longitudinal algorithms POC

This folder contains example inputs for running the following algorithms on
longitudinal datasets

- Linear regression
- Linear regression CV
- Logistic regression
- Logistic regression CV
- Anova one way
- Anova two way
- Gaussian naive Bayes CV

Run with

```bash
cat ltd_algos_input/<FILENAME>.json | ./run_algorithm -a generic_longitudinal
```
