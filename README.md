# K-Nearest Neighbors (KNN)

A machine learning project focused on understanding and implementing the **K-Nearest Neighbors (KNN)** algorithm for **Classification** and **Regression**.

The project combines mathematical understanding, custom implementation, dataset experiments, and visualization to build a clear understanding of how KNN works.

---

## 📌 Project Scope

The current project focuses on two main machine learning tasks:

1. **KNN Classification**
2. **KNN Regression**

The implementation is intentionally developed step by step instead of relying only on high-level machine learning libraries.

At the current stage, the project focuses on understanding:

* Euclidean distance
* Nearest-neighbor selection
* The effect of `K`
* Classification using majority voting
* Regression using the average value of neighbors
* Train/Test splitting
* Feature scaling
* Model evaluation
* Visualization of KNN predictions and decision boundaries

> Advanced neighbor-search algorithms and optimizations are **not part of the current scope** and will be considered in future development.

---

# 🧠 What is KNN?

**K-Nearest Neighbors (KNN)** is a supervised machine learning algorithm that makes predictions based on the closest training samples.

For a new data point:

1. Calculate the distance between the new point and training samples.
2. Find the `K` nearest samples.
3. Use those neighbors to make a prediction.

The distance currently used in the project is **Euclidean distance**.

For two points:

$$
x = (x_1, x_2, ..., x_n)
$$

and

$$
y = (y_1, y_2, ..., y_n)
$$

the Euclidean distance is:

$$
d(x,y) =
\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

---

# 🔵 KNN Classification

The Classification part of the project uses the **Breast Cancer Wisconsin dataset**.

The target variable is:

```text
diagnosis
```

where:

```text
B → 0
M → 1
```

The workflow includes:

```text
Dataset
   ↓
Data inspection
   ↓
Data cleaning
   ↓
Feature analysis
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Custom KNN
   ↓
Prediction
   ↓
Evaluation
```

## Current Classification Implementation

The project includes a custom KNN implementation:

```python
from CustomKNN import CustomKNN

knn = CustomKNN(k=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

The current implementation uses nearest neighbors and majority voting to determine the predicted class.

### Evaluation

The current notebook evaluates the classifier using:

* Accuracy
* Classification Report

Example:

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

# 🟠 KNN Regression

The Regression part of the project uses **S&P 500 stock data**, with the current experiment focusing on **AAL** stock.

The current experiment uses:

```text
open
high
low
```

as input features and:

```text
close
```

as the target value.

Therefore:

$$
X = [open, high, low]
$$

and:

$$
y = close
$$

The workflow is approximately:

```text
Stock Dataset
     ↓
Check Missing Values
     ↓
Data Cleaning
     ↓
Select AAL
     ↓
Select Last 6 Months
     ↓
Select Features
     ↓
Calculate Euclidean Distance
     ↓
Find K Nearest Neighbors
     ↓
Average Neighbor Values
     ↓
Prediction
     ↓
Visualization
```

For KNN Regression, the prediction is currently calculated as the average target value of the `K` nearest neighbors:

$$
\hat{y} =
\frac{1}{K}
\sum_{i=1}^{K} y_i
$$

Current implementation:

```python
k = 8

for i in range(N):
    k_indicates = np.argsort(D[i, :])[:k]

    y_pred[i] = np.sum(y[k_indicates]) / k
```

---

# 📊 Visualization

Visualization is used to understand how KNN behaves rather than treating the algorithm as a black box.

The project currently contains two main visualization directions.

## Classification Decision Boundary

The project generates a decision boundary to visualize how different regions of the feature space are classified.

Example concept:

```text
       Class 1
    ● ● ● ●
   ● ● ●

------------- Decision Boundary -------------

          ▲ ▲ ▲
        ▲ ▲ ▲ ▲
        Class 2
```

The `K` value directly affects the shape of the decision boundary.

A smaller `K` generally makes the prediction more dependent on nearby individual samples, while a larger `K` considers a broader neighborhood.

---

## Regression Visualization

The regression notebook visualizes:

* Stock price data
* KNN predicted values
* The relationship between the original data and predictions

The current experiment uses a candlestick visualization together with the KNN regression prediction.

---

# 🧪 Learning / Implementation Approach

This project does not only use `scikit-learn` as a black-box implementation.

Instead, the algorithm is also explored manually.

For example, the classification notebook `KNN_Classifier_V2.ipynb` explicitly performs:

### 1. Generate Dataset

```python
X = np.zeros((n, 2))
y = np.random.randint(0, clazz, n)
```

### 2. Calculate Distance

```python
D[i, j] = np.sqrt(
    (X[i, 0] - X[j, 0])**2 +
    (X[i, 1] - X[j, 1])**2
)
```

### 3. Find K Nearest Neighbors

```python
k_indicates = np.argsort(D[i, :])[:k]
```

### 4. Majority Voting

```python
y_candidates = y[k_indicates]
y_pred[i] = np.bincount(y_candidates).argmax()
```

This approach makes the internal mechanism of KNN explicit and easier to study.

---

# 📁 Project Notebooks

| Notebook                  | Purpose                                                              | Status         |
| ------------------------- | -------------------------------------------------------------------- | -------------- |
| `KNN_Classifier.ipynb`    | KNN Classification using Breast Cancer Wisconsin dataset             | ✅ Completed    |
| `KNN_Classifier_V2.ipynb` | Step-by-step KNN Classification implementation and decision boundary | ✅ Completed    |
| `KNN_Regression.ipynb`    | KNN Regression using stock market data                               | ✅ Completed    |
| `KNN_Visualize.ipynb`     | Visualization of KNN decision regions                                | 🚧 In Progress |

---

# 🚧 Project Status

The project will be developed incrementally.

| Feature / Topic                     |     Status     | Notes                              |
| ----------------------------------- | :------------: | ---------------------------------- |
| Understand KNN concept              |     ✅ Done     | Basic KNN workflow                 |
| Euclidean Distance                  |     ✅ Done     | Manually implemented               |
| KNN Classification                  |     ✅ Done     | Custom implementation              |
| Majority Voting                     |     ✅ Done     | Used for classification            |
| Train/Test Split                    |     ✅ Done     | Used in classification             |
| Feature Scaling                     |     ✅ Done     | `StandardScaler`                   |
| Classification Evaluation           |     ✅ Done     | Accuracy + Classification Report   |
| Decision Boundary                   |     ✅ Done     | Visualized for classification      |
| KNN Regression                      |     ✅ Done     | Custom implementation              |
| Neighbor Mean for Regression        |     ✅ Done     | Used for prediction                |
| Regression Visualization            | 🚧 In Progress | Current stock experiment           |
| Improve K selection                 |   📌 Planned   | Experiment with different K values |
| Compare different K values          |   📌 Planned   | Analyze model behavior             |
| Regression evaluation metrics       |   📌 Planned   | Further evaluation                 |
| Improve CustomKNN structure         |   📌 Planned   | Refactor implementation            |
| Distance optimization               |    📌 Future   | Improve computational efficiency   |
| Advanced neighbor-search algorithms |    📌 Future   | Not currently in scope             |
| `KDTree` / `BallTree`               |    📌 Future   | To be studied later                |
| Brute-force optimization            |    📌 Future   | To be studied later                |
| Automatic algorithm selection       |    📌 Future   | To be studied later                |

---

# 🗺️ Future Roadmap

The project will evolve gradually.

## Phase 1 — Core KNN

```text
KNN Concept
    ↓
Euclidean Distance
    ↓
Classification
    ↓
Regression
    ↓
Visualization
```

**Status: Mostly completed**

---

## Phase 2 — Model Analysis

Future work will focus on understanding how the model behaves under different configurations.

Possible topics:

* Different `K` values
* Effect of small vs. large `K`
* Classification performance
* Regression performance
* Model evaluation metrics
* Feature scaling experiments

---

## Phase 3 — Implementation Improvement

The custom KNN implementation can later be improved in terms of:

* Code structure
* Reusability
* Computational efficiency
* Distance calculation
* Prediction performance

---

## Phase 4 — Advanced Neighbor Search

Only after the core KNN implementation is well understood, the project may explore more advanced approaches for finding nearest neighbors.

Potential future topics include:

```text
Brute-force Search
       ↓
KDTree
       ↓
BallTree
       ↓
Automatic Algorithm Selection
```

These topics are intentionally **outside the current project scope**.

---

# 🎯 Current Goal

The main goal of the project is not to implement every feature available in a KNN library.

The current goal is to build a solid understanding of:

> **How KNN works internally and how it can be applied to both Classification and Regression problems.**

The project therefore prioritizes:

* Mathematical understanding
* Manual implementation
* Practical datasets
* Visualization
* Model evaluation
* Clear and incremental development

Advanced optimization and neighbor-search algorithms will be introduced later as the project progresses.
