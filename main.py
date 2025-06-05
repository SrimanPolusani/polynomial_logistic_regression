import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class LogisticRegression:
    def __init__(self, txt_file, alpha, num_iters, lambda_, degree=2, test_size=0.2, random_state=42):
        # Load original features and target from the entire dataset
        X_loaded_full = np.loadtxt(txt_file, usecols=(0, 1), delimiter=',')
        y_full = np.loadtxt(txt_file, usecols=2, delimiter=',')

        # Normalize original features based on the entire dataset
        self.X_mean_orig_full = np.mean(X_loaded_full, axis=0)
        self.X_std_orig_full = np.std(X_loaded_full, axis=0)
        self.X_std_orig_full[self.X_std_orig_full == 0] = 1.0
        X_normalized_full = (X_loaded_full - self.X_mean_orig_full) / self.X_std_orig_full

        # Create polynomial features from the entire normalized dataset
        self.degree = degree
        X_poly_full = self._map_features(X_normalized_full[:, 0], X_normalized_full[:, 1])

        # Split data into training and test sets (features and labels)
        self.X_train_orig_split, self.X_test_orig_split, \
            self.y_train, self.y_test = train_test_split(
            X_loaded_full, y_full, test_size=test_size, random_state=random_state,
            stratify=y_full if len(np.unique(y_full)) > 1 else None
        )

        # For training, we need the polynomial features corresponding to the training split
        # And normalized original features for plotting the training set later
        self.X_train_normalized_split = (self.X_train_orig_split - self.X_mean_orig_full) / self.X_std_orig_full
        self.X_train = self._map_features(self.X_train_normalized_split[:, 0], self.X_train_normalized_split[:, 1])

        # For plotting the test set later, we'll need its normalized original features
        self.X_test_normalized_split = (self.X_test_orig_split - self.X_mean_orig_full) / self.X_std_orig_full

        self.m_examples = self.X_train.shape[0]  # Number of training examples
        self.n_features = self.X_train.shape[1]  # Number of polynomial features

        self.learning_rate = alpha
        self.total_iters = num_iters
        self.reg_param = lambda_

        # Generate initial params
        np.random.seed(1)
        self.w = np.random.randn(self.n_features)
        self.b = np.random.randn()
        self.pos_train = self.y_train == 1
        self.neg_train = self.y_train == 0

        self.z_g_history = {}
        self.j_history = {}  # For cost vs iteration plotting
        self.min_cost_params_history = {}  # For finding best w,b

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

    # <-----Performing Gradient Descent----->
    def gradient_descent(self):
        cost_evolution_list = []  # Local list for storing cost history for plotting
        for iter_num in range(self.total_iters):
            w_gradient, b_gradient = self.compute_gradient()
            self.w = self.w - (self.learning_rate * w_gradient)
            self.b = self.b - (self.learning_rate * b_gradient)

            current_cost = self.compute_cost_logistic()
            cost_evolution_list.append(current_cost)  # Append to local list

            if iter_num < 100_000:
                self.min_cost_params_history[current_cost] = [self.w.copy(), self.b]

            if iter_num % math.ceil(self.total_iters / 10) == 0 or iter_num == self.total_iters - 1:
                print(f'Iteration {iter_num:6d}, Cost: {current_cost:.8f}')

        self.j_history = {"costs": cost_evolution_list}  # Store for plotting

        if self.min_cost_params_history:
            min_cost = min(self.min_cost_params_history.keys())
            self.w, self.b = self.min_cost_params_history[min_cost]
            print(f"\nLowest training cost found: {min_cost:.8f}, parameters updated.")
        return self.w, self.b, self.j_history

    def predict(self, X_orig_features_subset):
        # Normalize using full dataset's mean and std
        X_normalized_subset = (X_orig_features_subset - self.X_mean_orig_full) / self.X_std_orig_full
        # Map to polynomial features
        X_poly_subset = self._map_features(X_normalized_subset[:, 0], X_normalized_subset[:, 1])

        z = np.dot(X_poly_subset, self.w) + self.b
        sigmoid_output = 1 / (1 + np.exp(-z))
        return (sigmoid_output >= 0.5).astype(int)

    def evaluate(self, X_orig_features_subset, y_true_subset, dataset_name="Dataset"):
        print(f"\n--- Quantitative Evaluation on {dataset_name} Set ---")
        y_pred = self.predict(X_orig_features_subset)

        accuracy = accuracy_score(y_true_subset, y_pred)
        precision = precision_score(y_true_subset, y_pred, zero_division=0)
        recall = recall_score(y_true_subset, y_pred, zero_division=0)
        f1 = f1_score(y_true_subset, y_pred, zero_division=0)
        cm = confusion_matrix(y_true_subset, y_pred)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    # <-----z Vs g(z) Sigmoid Graph----->
    def visualize_sigmoid(self):
        if not self.z_g_history:
            print("Sigmoid history is empty. Run gradient descent first.")
            return
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        sorted_z = sorted(self.z_g_history.keys())
        sorted_g_z = [self.z_g_history[z_val] for z_val in sorted_z]
        axes.plot(sorted_z, sorted_g_z, color='#1AA7EC', markersize=5)
        axes.set_title('Sigmoid Function g(z)', fontsize=15)
        axes.set_ylabel('g(z)', fontsize=15)
        axes.set_xlabel('z', fontsize=15)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    # <-----Original Xi Vs Y_train Graphs----->
    def visualize_xygraph(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        X_to_plot = self.X_train_normalized_split
        y_to_plot = self.y_train
        pos_for_plot = y_to_plot == 1
        neg_for_plot = y_to_plot == 0

        for axis_num, axis in enumerate(axes):
            if X_to_plot.shape[1] > axis_num:
                x_data_pos = X_to_plot[pos_for_plot, axis_num]
                x_data_neg = X_to_plot[neg_for_plot, axis_num]
                axis.scatter(
                    x_data_pos, y_to_plot[pos_for_plot],
                    marker='x', s=80, c='red', label='y=1', lw=1
                )
                axis.scatter(
                    x_data_neg, y_to_plot[neg_for_plot],
                    marker='o', s=100, facecolors='none', edgecolors='blue', lw=1, label='y=0'
                )
                axis.set_title(f'Training Data: Normalized $X_{axis_num}$ Vs $Y_{{train}}$')
                axis.set_ylim(-0.08, 1.1)
                axis.set_xlabel(f'Training Data: Normalized $X_{axis_num}$')
                axis.set_ylabel('$Y_{train}$')
                axis.legend()
            else:
                axis.set_title(f'Original $X_{axis_num}$ N/A')
        plt.tight_layout()
        plt.show()

    # <-----X0 Vs X1 Graph and Polynomial Decision Curve----->
    def visualize_decision_boundary_on_set(self, X_normalized_subset, y_true_subset, dataset_name="Dataset"):
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))

        pos_subset = y_true_subset == 1
        neg_subset = y_true_subset == 0

        axes.scatter(
            X_normalized_subset[pos_subset, 0], X_normalized_subset[pos_subset, 1],
            marker='x', s=100, c='red', lw=2, label='Class 1'
        )
        axes.scatter(
            X_normalized_subset[neg_subset, 0], X_normalized_subset[neg_subset, 1],
            marker='o', s=80, facecolors='none', edgecolors='blue', lw=2, label='Class 0'
        )

        u_min, u_max = X_normalized_subset[:, 0].min() - 0.5, X_normalized_subset[:, 0].max() + 0.5
        v_min, v_max = X_normalized_subset[:, 1].min() - 0.5, X_normalized_subset[:, 1].max() + 0.5
        u = np.linspace(u_min, u_max, 100)
        v = np.linspace(v_min, v_max, 100)
        z_contour = np.zeros((len(u), len(v)))

        for i, u_val in enumerate(u):
            for j, v_val in enumerate(v):
                temp_x1_norm = np.array([u_val])
                temp_x2_norm = np.array([v_val])
                # These u_val, v_val are already on the normalized scale
                mapped_features_for_point = self._map_features(temp_x1_norm, temp_x2_norm)
                z_contour[i, j] = np.dot(mapped_features_for_point[0], self.w) + self.b

        Z_for_contour = z_contour.T
        axes.contour(u, v, Z_for_contour, levels=[0], colors='green', linewidths=2)

        axes.set_title(f'Decision Boundary on {dataset_name} Set (Degree {self.degree})', fontsize=16)
        axes.set_xlabel('Normalized Feature $X_0$', fontsize=14)
        axes.set_ylabel('Normalized Feature $X_1$', fontsize=14)
        axes.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()


# --- Main execution ---
# Load data
file_path = r"C:\Users\srima\PycharmProjects\logistic_regression\data2.txt"
X_orig_full = np.loadtxt(file_path, usecols=(0, 1), delimiter=',')
y_full = np.loadtxt(file_path, usecols=2, delimiter=',')

alph = 0.05
iters = 10000
lambda_value = 0.01
poly_degree = 3

ml_object = LogisticRegression(file_path, alph, iters, lambda_value, degree=poly_degree, test_size=0.2,
                               random_state=42)

print(f"Training set size: {ml_object.X_train_orig_split.shape[0]} samples")
print(f"Test set size: {ml_object.X_test_orig_split.shape[0]} samples")

# Train the model using gradient_descent (which uses internal training data)
final_w, final_b, cost_history_data = ml_object.gradient_descent()

# Visualize Cost History from training
if "costs" in cost_history_data:
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history_data["costs"])
    plt.title(f'Cost Function History (Degree {ml_object.degree})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(w,b)')
    plt.grid(True)
    plt.show()


ml_object.visualize_sigmoid()
ml_object.visualize_xygraph()

# --- Quantitative Evaluation and Visualization ---

# On Training Set
ml_object.evaluate(ml_object.X_train_orig_split, ml_object.y_train, dataset_name="Training")
ml_object.visualize_decision_boundary_on_set(ml_object.X_train_normalized_split, ml_object.y_train,
                                             dataset_name="Training")

# On Test Set
ml_object.evaluate(ml_object.X_test_orig_split, ml_object.y_test, dataset_name="Test")
ml_object.visualize_decision_boundary_on_set(ml_object.X_test_normalized_split, ml_object.y_test, dataset_name="Test")