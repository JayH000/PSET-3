import numpy as np
from scipy.integrate import nquad

# Define the integrand for computing moments
def moment_integrand(A, w, exponents):
    def integrand(*v):
        v = np.array(v)
        exponent = -0.5 * v.T @ A @ v + w.T @ v
        return np.exp(exponent) * np.prod(v**exponents)  # Multiply by v^p1, v^p2, etc.

    return integrand

# Function to compute numerical moments
def compute_numerical_moment(A, w, exponents):
    N = len(w)
    integrand = moment_integrand(A, w, exponents)
    integral_value, _ = nquad(integrand, [(-np.inf, np.inf)] * N)
    
    # Normalize integral 
    normalization, _ = nquad(lambda *v: np.exp(-0.5 * np.dot(v, A @ v) + np.dot(w, v)), [(-np.inf, np.inf)] * N)
    
    return integral_value / normalization

# compute analytical moments
def compute_analytical_moments(A, w):
    A_inv = np.linalg.inv(A)
    mean_vector = A_inv @ w  # <v>
    
    second_moments = np.outer(mean_vector, mean_vector) + A_inv  # <vi vj>
    
    third_moments = {}
    third_moments[(0, 0, 1)] = mean_vector[0]**2 * mean_vector[1] + 2 * mean_vector[0] * A_inv[0, 1] + A_inv[0, 0] * mean_vector[1]
    third_moments[(1, 2, 2)] = mean_vector[1] * mean_vector[2]**2 + 2 * mean_vector[2] * A_inv[1, 2] + A_inv[2, 2] * mean_vector[1]

    fourth_moments = {}
    fourth_moments[(0, 0, 1, 1)] = mean_vector[0]**2 * mean_vector[1]**2 + 2 * A_inv[0, 0] * mean_vector[1]**2 + 2 * A_inv[1, 1] * mean_vector[0]**2 + 4 * mean_vector[0] * mean_vector[1] * A_inv[0, 1] + 2 * A_inv[0, 1]**2

    return mean_vector, second_moments, third_moments, fourth_moments

# Matrices and Vector 
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])  # Positive definite
w = np.array([1, 2, 3])

# Compute numerical and analytical moments
numerical_moments = {
    "E[v1]": compute_numerical_moment(A, w, [1, 0, 0]),
    "E[v2]": compute_numerical_moment(A, w, [0, 1, 0]),
    "E[v3]": compute_numerical_moment(A, w, [0, 0, 1]),
    "E[v1v2]": compute_numerical_moment(A, w, [1, 1, 0]),
    "E[v2v3]": compute_numerical_moment(A, w, [0, 1, 1]),
    "E[v1v3]": compute_numerical_moment(A, w, [1, 0, 1]),
    "E[v1^2 v2]": compute_numerical_moment(A, w, [2, 1, 0]),
    "E[v2 v3^2]": compute_numerical_moment(A, w, [0, 1, 2]),
    "E[v1^2 v2^2]": compute_numerical_moment(A, w, [2, 2, 0]),
    "E[v2^2 v3^2]": compute_numerical_moment(A, w, [0, 2, 2]),
}

analytical_moments = compute_analytical_moments(A, w)

# Print results
print("Numerical Moments:")
for key, value in numerical_moments.items():
    print(f"{key}: {value}")

print("\nAnalytical Moments:")
print(f"E[v1], E[v2], E[v3]: {analytical_moments[0]}")
print(f"E[v1 v2], E[v2 v3], E[v1 v3]:\n{analytical_moments[1]}")
print(f"E[v1^2 v2]: {analytical_moments[2].get((0, 0, 1), 'N/A')}")
print(f"E[v2 v3^2]: {analytical_moments[2].get((1, 2, 2), 'N/A')}")
print(f"E[v1^2 v2^2]: {analytical_moments[3].get((0, 0, 1, 1), 'N/A')}")
