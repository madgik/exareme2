import numpy as np


def get_model_parameters(model):
    params = [model.coef_]
    if model.fit_intercept:
        params.append(model.intercept_)
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]


def set_initial_params(model, X_train, full_data, flower_inputdata):
    model.classes_ = np.array(
        [i for i in range(len(np.unique(full_data[flower_inputdata.y])))]
    )
    model.coef_ = np.zeros((len(model.classes_), X_train.shape[1]))
    if model.fit_intercept:
        model.intercept_ = np.zeros((len(model.classes_),))
