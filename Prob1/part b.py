import numpy as np
from scipy.integrate import nquad

def gaussian_integral_verification(A, w):
    N = len(w)  # Dimensionality
    
    # Define the integrand
    def integrand(*v):
        v = np.array(v)
        exponent = -0.5 * v.T @ A @ v + w.T @ v
        return np.exp(exponent)
    
    # Compute the integral using numerical quadrature
    integral_value, _ = nquad(integrand, [(-np.inf, np.inf)] * N)
    
    # Compute the closed-form solution
    A_inv = np.linalg.inv(A)
    determinant_term = np.sqrt((2 * np.pi) ** N * np.linalg.det(A_inv))
    exponent_term = np.exp(0.5 * w.T @ A_inv @ w)
    closed_form_value = determinant_term * exponent_term
    
    # Compare results
    return integral_value, closed_form_value, np.isclose(integral_value, closed_form_value, rtol=1e-3)

# Given Matrices and Vector
A1 = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])  # Positive definite
A2 = np.array([[4, 2, 1], [2, 1, 3], [1, 3, 6]])  # Might not be positive definite
w = np.array([1, 2, 3])

# Run test cases
numerical_A1, closed_A1, verified_A1 = gaussian_integral_verification(A1, w)
numerical_A2, closed_A2, verified_A2 = gaussian_integral_verification(A2, w)

# Print results for A
print("Results for A:")
print(f"Numerical Integral: {numerical_A1}")
print(f"Closed-form Result: {closed_A1}")
print(f"Verification: {'Success' if verified_A1 else 'Failure'}\n")

# Print results for A'
print("Results for A':")
print(f"Numerical Integral: {numerical_A2}")
print(f"Closed-form Result: {closed_A2}")
print(f"Verification: {'Success' if verified_A2 else 'Failure'}")

# Check if A' is positive definite
eigvals_A2 = np.linalg.eigvals(A2)
print("\nEigenvalues of A':", eigvals_A2)
print("Is A' positive definite?", np.all(eigvals_A2 > 0))
