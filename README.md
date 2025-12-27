# Logistic Regression from Scratch — Polynomial Features + L2 Regularization

---

## 1. Results Visualization
<table>
   <tr>
    <td align="center">
      <img src="https://github.com/SrimanPolusani/polynomial-logistic-regression/blob/master/result_plots/decision_curve_train_plot.png?raw=true" width="100%" />
      <br />
      <b>Decision Boundary (Training)</b>
    </td>
    <td align="center">
      <img src="https://github.com/SrimanPolusani/polynomial-logistic-regression/blob/master/result_plots/decision_curve_test_plot.png?raw=true" width="100%" />
      <br />
      <b>Decision Boundary (Test)</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/SrimanPolusani/polynomial-logistic-regression/blob/master/result_plots/cost_func_plot.png?raw=true" width="100%" />
      <br />
      <b>Cost vs Iterations</b>
    </td>
    <td align="center">
      <img src="https://github.com/SrimanPolusani/polynomial-logistic-regression/blob/master/result_plots/sigmoid_func_plot.png?raw=true" width="100%" />
      <br />
      <b>Sigmoid g(z) Samples</b>
    </td>
  </tr>
</table>
<table>
  <tr>
      <td align="center">
        <img src="https://github.com/SrimanPolusani/polynomial-logistic-regression/blob/master/result_plots/confusion_matrix.png" />
        <br />
        <b>Confusion Matrix</b>
      </td>
    </tr>
</table>


---

## 2. Quantitative Results

The model was trained for **10,000 iterations** with a learning rate of **0.05** and regularization lambda of **0.01**.

### Performance Metrics

<table>
  <thead>
    <tr>
      <th align="left">Metric</th>
      <th align="center">Training Set</th>
      <th align="center">Test Set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><b>Accuracy</b></td>
      <td align="center">0.8511</td>
      <td align="center">0.8333</td>
    </tr>
    <tr>
      <td align="left"><b>Precision</b></td>
      <td align="center">0.8478</td>
      <td align="center">0.7857</td>
    </tr>
    <tr>
      <td align="left"><b>Recall</b></td>
      <td align="center">0.8478</td>
      <td align="center">0.9167</td>
    </tr>
    <tr>
      <td align="left"><b>F1-score</b></td>
      <td align="center">0.8478</td>
      <td align="center">0.8462</td>
    </tr>
  </tbody>
</table>

## 3. Model Analysis
The model demonstrates strong generalization with minimal variance between train and test scores (approx. 1.8% drop).
* **Decision Boundary:** The polynomial features successfully captured the non-linear, circular distribution of the data, which a linear model would have missed.
* **Error Analysis:** As seen in the Test Confusion Matrix, the model is highly effective at identifying the positive class (Recall: ~91.7%), with only **1 False Negative**. However, it produced **3 False Positives**, resulting in a slightly lower Precision (~78.6%).

---
## 4. Deep Dive: Logistic Regression Implementation from Scratch
### 4.1. Imports and Dependencies
```python

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
```

Imports the necessary libraries for mathematical operations (`numpy`), data visualization (`matplotlib`), and mathematical functions (`math`). It also imports `sklearn` utilities for data splitting and model evaluatation metrics (Accuracy, Precision, Recall, F1).

---

### 4.2. Class Initialization and Data Loading

```python
class LogisticRegression:
    def __init__(self, txt_file, alpha, num_iters, lambda_, degree=2, test_size=0.2, random_state=42):
        # Load original features and target from the entire dataset
        X_loaded_full = np.loadtxt(txt_file, usecols=(0, 1), delimiter=',')
        y_full = np.loadtxt(txt_file, usecols=2, delimiter=',')

        # 2. Split data
        self.X_train_orig_split, self.X_test_orig_split, \
            self.y_train, self.y_test = train_test_split(
            X_loaded_full, y_full, test_size=test_size, random_state=random_state,
            stratify=y_full if len(np.unique(y_full)) > 1 else None
        )

```

The `__init__` method initializes the model. It loads data from a text file where the first two columns are features ($x_1, x_2$) and the third is the target label $y$.
it uses `train_test_split` to reserve 20% (`test_size=0.2`) of the data for testing. The `stratify` parameter ensures that the proportion of Class 0 and Class 1 remains the same in both the training and test sets.

---

### 4.3. Feature Scaling (Normalization)

```python
        # 3. Calculate mean/std on the training Set
        self.X_mean = np.mean(self.X_train_orig_split, axis=0)
        self.X_std = np.std(self.X_train_orig_split, axis=0)
        self.X_std[self.X_std == 0] = 1.0

        # 4. Normalize the training set
        self.X_train_normalized_split = (self.X_train_orig_split - self.X_mean) / self.X_std

        # 5. Normalize the test set using training stats
        self.X_test_normalized_split = (self.X_test_orig_split - self.X_mean) / self.X_std

```

This section performs **Z-score Normalization**. Normalization is critical for Gradient Descent to converge faster. If features have vastly different scales (e.g., house size in feet vs. number of bedrooms), the cost function contours become elongated ovals, making optimization slow. Scaling makes them circular.

The test set is normalized using the *training set's* mean and standard deviation to prevent "data leakage".

**Mean and Standard Deviation for feature $j$:**

$$\large
\mu_j = \frac{1}{m} \sum_{i=1}^m x^{(i)}_j
$$

$$\large
\sigma_j = \sqrt{ \frac{1}{m} \sum_{i=1}^m (x^{(i)}_j - \mu_j)^2 }
$$

**The z-score normalized feature $x'$ is:**

$$\large
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$

**Where:**
* $\mu_j$: The mean (average) of feature $j$.
* $\sigma_j$: The standard deviation of feature $j$.
* $m$: The total number of training examples.
* $x^{(i)}_j$: The value of feature $j$ in the $i$-th training example.
* $x'_j$: The resulting normalized value.
---

### 4.4. Polynomial Feature Mapping and Initialization

```python
        # Create polynomial features using the normalized training split
        self.degree = degree
        self.X_train = self._map_features(self.X_train_normalized_split[:, 0], self.X_train_normalized_split[:, 1])

        self.m_examples = self.X_train.shape[0]  # Number of training examples
        self.n_features = self.X_train.shape[1]  # Number of polynomial features

        self.learning_rate = alpha
        self.total_iters = num_iters
        self.reg_param = lambda_

        # Generate initial params
        np.random.seed(1)
        self.w = np.random.randn(self.n_features)
        self.b = np.random.randn()

```

Here, the code maps the original 2 features into a higher-dimensional space (determined by `degree`) to allow the model to fit non-linear decision boundaries.

It also initializes the weights vector $\mathbf{w}$ and bias scalar $b$ randomly. `self.reg_param` stores the regularization parameter $\lambda$ (lambda).

**Linear Model Equation:**

$$\large
z = \mathbf{w} \cdot \mathbf{x} + b
$$

**Where:**
* $z$: The linear output (also called the log-odds or weighted sum).
* $\mathbf{w}$: The weight vector (parameters determining the importance of each feature).
* $\mathbf{x}$: The input feature vector.
* $b$: The bias term (intercept).

---

### 4.5. Feature Mapping Helper

```python
    def _map_features(self, X1, X2):
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        out_features = []
        for i in range(1, self.degree + 1):
            for j in range(i + 1):
                term = (X1 ** (i - j)) * (X2 ** j)
                out_features.append(term)
        if not out_features:
            return np.hstack((X1, X2))
        return np.hstack(out_features)

```

This function generates polynomial features. If `degree=2`, it creates terms like $x_1, x_2, x_1^2, x_1x_2, x_2^2$. This transformation allows Logistic Regression (a linear classifier) to separate classes that require a curve (like a circle) to divide them.

**The feature set becomes:**

$$\large
\phi(x) = [x_1, x_2, x_1^2, x_1 x_2, x_2^2, x_1^3, \dots ]
$$

**Where:**
* $\phi(x)$: The expanded feature vector (Phi of x) containing the new polynomial terms.
* $x_1, x_2$: The original input features
* $x_1^2, x_2^2$: The squared terms, allowing for circular/elliptical decision boundaries.
* $x_1 x_2$: The interaction term, capturing the relationship between the two features.

---

### 4.6. Cost Function (Logistic Loss + Regularization)

```python
    # <-----Cost Calculation----->
    def compute_cost_logistic(self):  # Operates on self.X_train and self.y_train
        cost = 0
        for ex_index in range(self.m_examples):  # m_examples is training examples
            zi = np.dot(self.w, self.X_train[ex_index]) + self.b
            sigmoid = 1 / (1 + np.exp(-zi))
            self.z_g_history[zi] = sigmoid

            cost_one = -self.y_train[ex_index] * np.log(sigmoid + 1e-10)
            cost_two = (1 - self.y_train[ex_index]) * np.log(1 - sigmoid + 1e-10)
            cost += cost_one - cost_two
        cost = cost / self.m_examples

        reg_part = 0
        for j in range(self.n_features):
            reg_part += self.w[j] ** 2
        reg_part = reg_part * (self.reg_param / (self.m_examples * 2.0))
        cost = cost + reg_part
        return cost

```

This computes the **Regularized Binary Cross-Entropy Cost**.

* **Sigmoid Activation:** Converts the linear output $z$ into a probability between 0 and 1.
* **Log Loss:** Penalizes the model heavily if it predicts probability $\approx 0$ when the actual label is 1 (and vice versa). `1e-10` is added to avoid `log(0)` errors.
* **Regularization:** The term `reg_part` adds a penalty based on the magnitude of the weights. This prevents overfitting by keeping weights small, which smooths the decision boundary.

**Sigmoid Activation Function:**

$$\large
f_{\mathbf{w},b}(\mathbf{x}) = g(z) = \frac{1}{1 + e^{-z}}
$$

**Regularized Cost Function** $J(\mathbf{w},b)$:

$$\large
J(\mathbf{w},b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{\mathbf{w},b}(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - f_{\mathbf{w},b}(\mathbf{x}^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

**Where:**
* $f_{\mathbf{w},b}(\mathbf{x})$: The model's prediction (hypothesis), representing the probability that $y=1$.
* $g(z)$: The sigmoid function, which maps any real number to the range $(0, 1)$.
* $e$: Euler's number (approx. 2.718).
* $J(\mathbf{w},b)$: The total cost (error) calculated over all training examples.
* $m$: The total number of training examples.
* $y^{(i)}$: The actual target label (0 or 1) for the $i$-th example.
* $\lambda$: The regularization parameter (controls how much to penalize large weights).
* $\sum w_j^2$: The regularization term (L2 penalty) that prevents overfitting.


---

### 4.7. Gradient Computation

```python
    # <-----dj_dw, dj_db Calculation----->
    def compute_gradient(self):  # Operates on self.X_train and self.y_train
        dj_dw = np.zeros(self.n_features)
        dj_db = 0
        for ex_index in range(self.m_examples):  # m_examples is training examples
            zi = np.dot(self.w, self.X_train[ex_index]) + self.b
            sigmoid = 1 / (1 + np.exp(-zi))
            err = sigmoid - self.y_train[ex_index]
            for feature_num in range(self.n_features):
                dj_dw[feature_num] = dj_dw[feature_num] + (err * self.X_train[ex_index, feature_num])
            dj_db = dj_db + err
        dj_dw = dj_dw / self.m_examples
        dj_db = dj_db / self.m_examples
        dj_dw += (self.reg_param / self.m_examples) * self.w
        return dj_dw, dj_db

```
This calculates the partial derivatives of the cost function with respect to weights and bias. These gradients point in the direction of the steepest ascent; we will subtract them to minimize cost.

Note that the regularization term $\frac{\lambda}{m} w_j$ is added to `dj_dw` but **not** to `dj_db` (we do not regularize the bias term).


Gradient for weights $\mathbf{w}$ (for feature $j$):

$$\large
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m} w_j
$$

**Gradient for bias $b$:**

$$\large
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
$$

**Where:**
* $\frac{\partial J}{\partial w_j}$: The gradient of the cost function with respect to weight $j$.
* $m$: The total number of training examples.
* $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}$: The prediction error (Prediction minus Actual).
* $x_j^{(i)}$: The value of feature $j$ in the $i$-th training example.
* $\frac{\lambda}{m} w_j$: The derivative of the regularization term (pushes weights toward zero).


---

### 4.8. Gradient Descent Optimization

```python
    # <-----Performing Gradient Descent----->
    def gradient_descent(self):
        cost_evolution_list = []  # Local list for storing cost history for plotting
        for iter_num in range(self.total_iters):
            w_gradient, b_gradient = self.compute_gradient()
            self.w = self.w - (self.learning_rate * w_gradient)
            self.b = self.b - (self.learning_rate * b_gradient)

            current_cost = self.compute_cost_logistic()
            cost_evolution_list.append(current_cost)
            # ... (logging and history tracking code)
        return self.w, self.b, self.j_history

```

This is training loop. In every iteration, it:
1.  **Computes gradients:** Calculates the direction in which the cost increases.
2.  **Updates parameters:** Adjusts $\mathbf{w}$ and $b$ by moving *against* the gradient by a step size `alpha` (learning rate).
3.  **Tracks the cost:** Monitors the cost to ensure it is decreasing over time.


**Parameter update rules:**

$$\large
w_j = w_j - \alpha \frac{\partial J}{\partial w_j}
$$

$$\large
b = b - \alpha \frac{\partial J}{\partial b}
$$

**Where:**
* $w_j$: The weight associated with feature $j$.
* $b$: The bias term.
* $\alpha$: The learning rate (Alpha).
* $\frac{\partial J}{\partial w_j}$: The gradient (slope) of the cost function. We subtract this because we want to go *down* the slope to minimize error.


---

### 4.9. Prediction

```python
    def predict(self, X_orig_features_subset):
        # Normalize using full dataset's mean and std
        X_normalized_subset = (X_orig_features_subset - self.X_mean) / self.X_std
        # Map to polynomial features
        X_poly_subset = self._map_features(X_normalized_subset[:, 0], X_normalized_subset[:, 1])

        z = np.dot(X_poly_subset, self.w) + self.b
        sigmoid_output = 1 / (1 + np.exp(-z))
        return (sigmoid_output >= 0.5).astype(int)

```

This method takes raw input features, normalizes them (using the training stats), maps them to polynomial features, and computes the probability.

It applies a threshold of $0.5$. If the probability $\geq 0.5$, it predicts Class 1; otherwise, it predicts Class 0.



$$\large
\hat{y} = \begin{cases} 
1 & \text{if } g(z) \geq 0.5 \\\\
0 & \text{if } g(z) < 0.5 
\end{cases}
$$

**Where:**
* $\hat{y}$: The final predicted class label (0 or 1).
* $g(z)$: The output of the sigmoid function, representing the probability $P(y=1 | x)$.
* $0.5$: The decision threshold. Probabilities above this are classified as positive; below are classified as negative.


---

### 4.10. Evaluation Metrics

```python
    def evaluate(self, X_orig_features_subset, y_true_subset, dataset_name="Dataset"):
        print(f"\n--- Quantitative Evaluation on {dataset_name} Set ---")
        y_pred = self.predict(X_orig_features_subset)

        accuracy = accuracy_score(y_true_subset, y_pred)
        precision = precision_score(y_true_subset, y_pred, zero_division=0)
        recall = recall_score(y_true_subset, y_pred, zero_division=0)
        f1 = f1_score(y_true_subset, y_pred, zero_division=0)
        cm = confusion_matrix(y_true_subset, y_pred)
        # ... (print statements)

```

This calculates quantitative metrics to assess the model:

* **Accuracy:** Overall percentage of correct predictions.
* **Precision:** Of all predicted positives, how many were actually positive?
* **Recall:** Of all actual positives, how many did we find?
* **F1-Score:** Harmonic mean of Precision and Recall. Useful when classes are imbalanced.

$$\large
\text{Precision} = \frac{TP}{TP + FP}
$$

$$\large
\text{Recall} = \frac{TP}{TP + FN}
$$

$$\large
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Where:**
* $TP$: True Positives
* $FP$: False Positives 
* $FN$: False Negatives

---

### 4.11. Decision Boundary Visualization

```python
    def visualize_decision_boundary_on_set(self, X_normalized_subset, y_true_subset, dataset_name="Dataset"):
        # ... (Scattering points) ...
        u = np.linspace(u_min, u_max, 100)
        v = np.linspace(v_min, v_max, 100)
        z_contour = np.zeros((len(u), len(v)))

        for i, u_val in enumerate(u):
            for j, v_val in enumerate(v):
                # ... (Map features for grid points) ...
                z_contour[i, j] = np.dot(mapped_features_for_point[0], self.w) + self.b

        Z_for_contour = z_contour.T
        axes.contour(u, v, Z_for_contour, levels=[0], colors='green', linewidths=2)

```

To visualize the non-linear boundary, the code performs the following steps:
1.  **Grid Creation:** Creates a grid of points ($u$ and $v$) covering the entire plot area.
2.  **Calculation:** Computes the value of the linear model $z$ for every point in that grid.
3.  **Contour Plotting:** Uses `axes.contour` to draw a line specifically where $z=0$.

**Why $z=0$?**
The sigmoid function output is exactly $0.5$ when the input $z$ is $0$. Therefore, this contour represents the "tipping point" between Class 0 and Class 1.



**The decision boundary:**

$$\large
z = \mathbf{w} \cdot \phi(x) + b = 0
$$

**Corresponding to the probability threshold:**

$$\large
g(z) = \frac{1}{1 + e^{-0}} = 0.5
$$

**Where:**
* $u, v$: The coordinates for the grid points used to map the decision surface.
* $z=0$: The decision boundary line. On one side, values are positive (Class 1); on the other, negative (Class 0).
* $P(y=1) > 0.5$: The region where the model predicts the positive class.
* $\phi(x)$: The polynomial feature mapping that allows the boundary to be curved (non-linear).

---

## 5. Main Execution

```python
# --- Main execution ---
file_path = r"C:\...\data2.txt"
# ...
ml_object = LogisticRegression(file_path, alph, iters, lambda_value, degree=poly_degree, test_size=0.2, random_state=42)

# Train the model
final_w, final_b, cost_history_data = ml_object.gradient_descent()

# Visualize Cost History
# ...
ml_object.evaluate(ml_object.X_test_orig_split, ml_object.y_test, dataset_name="Test")

```

This acts as the driver code. It sets hyperparameters (Alpha, Lambda, Degree), instantiates the class, runs gradient descent, and finally triggers the evaluation and visualization methods on both training and test sets to verify the model works.

```

```
