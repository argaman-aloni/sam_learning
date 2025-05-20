import cvxpy as cp
import numpy as np

#
# def polyhedral_separation(positive_samples: np.array, negative_samples: np.array, h: int):
#     """
#     Polyhedral separation of positive and negative samples.
#
#     Args:
#         positive_samples: numpy array of shape (n_samples, n_features)
#         negative_samples: numpy array of shape (n_samples, n_features)
#         w: numpy array of shape (n_features,)
#         eps: tolerance for separation
#     """
#     w_var = cp.Variable(shape=(h, positive_samples.shape[1]))
#     b_var = cp.Variable(shape=(h, 1))
#
#     # The constraints are as follows: (1) for each positive sample x_i, and for any hyperplane vector defined by w_j and b_j:, x_i^T @ w_j + b_j <= -1
#     # (2) for each negative sample x_i, and there exist a hyperplane vector defined by w_j and b_j such that:, x_i^T @ w_j + b_j >= 1
#     constraints = []
#
#     # for each positive sample x_i, and for any hyperplane vector defined by w_j and b_j:, x_i^T @ w_j + b_j <= -1
#     for i, x in enumerate(positive_samples):
#         constraints.append(x.T @ w_var[i] + b_var[i] <= -1)
#
#     # for each negative sample x_i, and there exist a hyperplane vector defined by w_j and b_j such that:, x_i^T @ w_j + b_j >= 1
#     # we express this as: max_j (x^T w + b_j) >= 1
#     for i, x in enumerate(negative_samples):
#         constraints.append(cp.max(x.T @ w_var[i] + b_var[i]) >= 1)

# for i, x in enumerate(positive_samples):
#     max_constraint = cp.max(x.T @ w_var[i] + b_var[i])
#
# # positive_sample_loss =
# negative_sample_loss = (1 / negative_samples.shape[0]) * cp.sum(cp.pos(cp.minimum([(1 - x @ w_var - b_var)])))
# loss_function = cp.sum(cp.pos(positive_samples @ w_var + b_var)) + cp.sum(cp.pos(cp.minimum()))


from ortools.linear_solver import pywraplp


def print_hyperplanes(W, b, sign="<=", rhs_value=-1):
    """
    Pretty print the hyperplane inequalities.

    Parameters:
    - W: numpy array of shape (h, n), weight vectors for h hyperplanes
    - b: numpy array of shape (h,), bias terms
    - sign: string, "<=" or ">=" depending on which side of the margin you want
    - rhs_value: int or float, the right-hand side of the inequality
    """
    h, n = W.shape
    for j in range(h):
        terms = [f"{W[j, i]:+.4f}*x{i}" for i in range(n)]
        equation = " ".join(terms)
        equation += f" {b[j]:+.4f}"
        print(f"Hyperplane {j+1}: {equation} {sign} {rhs_value}")


# Toy dataset
X_plus = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0], [10.0, 0.0]])
X_minus = np.array([[-70.0, -80.0], [80.0, 90.0], [70.0, 90.0], [80.0, 80.0]])
X = np.vstack([X_plus, X_minus])
y = np.array([1] * len(X_plus) + [-1] * len(X_minus))

m = len(X_plus)
p = len(X)
n = X.shape[1]
h = 4
M = 1000  # Big-M

# Create solver
solver = pywraplp.Solver.CreateSolver("SCIP")

# Variables w[j][k] and b[j]
w = [[solver.NumVar(-solver.infinity(), solver.infinity(), f"w_{j}_{k}") for k in range(n)] for j in range(h)]
b = [solver.NumVar(-solver.infinity(), solver.infinity(), f"b_{j}") for j in range(h)]

# Binary variables z[i][j] for negative points
z = [[solver.IntVar(0, 1, f"z_{i}_{j}") for j in range(h)] for i in range(m, p)]

# Positive constraints
for i in range(m):
    for j in range(h):
        solver.Add(sum(X[i][k] * w[j][k] for k in range(n)) + b[j] <= -1)

# Negative constraints with Big-M
for idx, i in enumerate(range(m, p)):
    solver.Add(solver.Sum(z[idx][j] for j in range(h)) >= 1)
    for j in range(h):
        solver.Add(sum(X[i][k] * w[j][k] for k in range(n)) + b[j] >= 1 - M * (1 - z[idx][j]))

# Solve (feasibility)
solver.Minimize(0)
status = solver.Solve()

# Output result
if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
    W_result = np.array([[w[j][k].solution_value() for k in range(n)] for j in range(h)])
    b_result = np.array([b[j].solution_value() for j in range(h)])
    print("W =", W_result)
    print("b =", b_result)
else:
    print("No solution found.")


print_hyperplanes(W_result, b_result, sign="<=", rhs_value=-1)  # For positive class


if __name__ == "__main__":
    pass
