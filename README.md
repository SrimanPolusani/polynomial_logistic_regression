# Logistic Regression from Scratch — Polynomial Features + L2 Regularization

---

## 1. Results Visualization
<table>
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
## 4. Deep Dive

