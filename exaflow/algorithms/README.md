# Creating Tests

In order to test the validity of the algorithms, we need to test specific parts of the implementation. In brief:

### Client Algorithm Testing:

- **Training Functionality**: Ensure the algorithm correctly updates its parameters or state given a batch of data.
- **Prediction Functionality**: Test the model’s ability to make predictions on new, unseen data.
- **Evaluation Functionality**: Verify the algorithm can calculate correct metrics (like loss, accuracy, F1 score, etc.) on a validation set.

### Server Algorithm Testing:

- **Model Aggregation**: Test the server’s ability to correctly aggregate models or parameters from multiple clients.
- **Configuration Management**: Ensure the server correctly sets up rounds and other configuration settings based on the learning task.

### Integration Testing:

- **End-to-End Functionality**: Validate the complete cycle from data distribution, client training, parameter updates, server aggregation, and evaluation across multiple rounds.
- **Robustness to Diverse Data**: Test how the algorithm handles different kinds of data distributions across clients.

______________________________________________________________________

Following, there is a general example of how we would test any algorithm. Bear in mind, this is an example that serves to guide the developer to create a similar example, using mostly the logic of the test, not the code. The purpose is to outline the major areas that should be tested in each algorithm to ensure the algorithm developed will yield the proper results without retesting robust components.

Note: The code blocks below are illustrative pseudo-code. Replace the placeholder client/server imports with the actual modules under `exaflow/algorithms/flower/<algorithm>/` (for example, `logistic_regression_fedaverage_flower/client.py` and `server.py`).

______________________________________________________________________

## Test Client (Generic)

```python

import pytest
import numpy as np
from sklearn.base import BaseEstimator
# Replace these placeholders with the algorithm-specific Flower client and utils.
from your_algorithm.client import YourAlgorithmClient
from your_algorithm.utils import get_model_parameters, set_model_params


@pytest.fixture
def sample_data():
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, size=100)
    return X_train, y_train


def test_client_training(sample_data):
    X_train, y_train = sample_data
    model = BaseEstimator()  # Replace with actual model class
    client = YourAlgorithmClient(model, X_train, y_train)

    parameters = client.get_parameters()
    new_parameters, num_examples, _ = client.fit(parameters, {})

    assert not np.array_equal(parameters, new_parameters), "Parameters should be updated after training"


def test_client_evaluation(sample_data):
    X_train, y_train = sample_data
    model = BaseEstimator()  # Replace with actual model class
    client = YourAlgorithmClient(model, X_train, y_train)

    parameters = client.get_parameters()
    loss, num_examples, metrics = client.evaluate(parameters, {})

    assert loss >= 0, "Loss should be non-negative"
    assert "accuracy" in metrics, "Metrics should include accuracy"

```

## Test Server (Generic)

```python

import pytest
from unittest.mock import patch
from sklearn.base import BaseEstimator
# Replace these placeholders with the algorithm-specific Flower server functions.
from your_algorithm.server import get_evaluate_fn, configure_round


def test_server_aggregation():
    model = BaseEstimator()  # Replace with actual model class
    evaluate_fn = get_evaluate_fn(model)

    parameters = [np.zeros(10), np.zeros(1)]  # Mock parameters
    loss, metrics = evaluate_fn(1, parameters, {})

    assert loss >= 0, "Loss should be non-negative"
    assert "accuracy" in metrics, "Metrics should include accuracy"


def test_round_configuration():
    config = configure_round(1)
    assert isinstance(config, dict), "Configuration should be a dictionary"

```

# General Testing Methodology

## Tests Setup

```python

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator

# Assuming `AlgorithmClass` is your custom class for the algorithm
from your_algorithm_library import AlgorithmClass

@pytest.fixture(scope="module")
def data_fixture():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    return X, y

@pytest.fixture(scope="module")
def model_fixture():
    return AlgorithmClass(parameters...)
```

## Test Cases

```python

def test_training(model_fixture, data_fixture):
    X, y = data_fixture
    model = model_fixture
    model.train(X, y)
    assert model.is_trained, "The model should be marked as trained."

def test_prediction(model_fixture, data_fixture):
    X, y = data_fixture
    model = model_fixture
    predictions = model.predict(X)
    assert len(predictions) == len(y), "The number of predictions should match the number of samples."

def test_accuracy(model_fixture, data_fixture):
    X, y = data_fixture
    model = model_fixture
    model.train(X, y)
    accuracy = model.evaluate(X, y)
    assert accuracy > 0.5, "Accuracy should be greater than 50% for a binary classification."

def test_serialization(model_fixture):
    model = model_fixture
    serialized = model.serialize()
    new_model = AlgorithmClass.deserialize(serialized)
    assert new_model.get_params() == model.get_params(), "Deserialized model should have the same parameters."

def test_large_dataset(model_fixture):
    X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
    model = model_fixture
    model.train(X, y)
    assert model.is_trained, "The model should handle large datasets without failure."

def test_noisy_data(model_fixture, data_fixture):
    X, y = data_fixture
    X_noisy = X + np.random.normal(0, 1, X.shape)
    model = model_fixture
    model.train(X_noisy, y)
    accuracy = model.evaluate(X_noisy, y)
    assert accuracy < 0.8, "Expect lower accuracy on noisy data, but it should still perform reasonably."

@pytest.mark.parametrize("n_classes, expected", [
    (2, True),
    (1, False),
    (10, True)
])
def test_varied_class_sizes(n_classes, expected):
    X, y = make_classification(n_samples=100, n_features=20, n_classes=n_classes, random_state=42)
    model = AlgorithmClass()
    try:
        model.train(X, y)
        result = True
    except Exception:
        result = False
    assert result == expected, f"Model should handle {n_classes} classes as expected."
```
