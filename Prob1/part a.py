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

# Example usage
A = np.array([[2.0, 0.5], [0.5, 1.5]])  # Symmetric positive-definite matrix
w = np.array([1.0, -1.0])  # Example vector

numerical_result, closed_form_result, is_verified = gaussian_integral_verification(A, w)
print(f"Numerical Integral: {numerical_result}")
print(f"Closed-form Result: {closed_form_result}")
print(f"Verification: {'Success' if is_verified else 'Failure'}")
