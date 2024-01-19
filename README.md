### **Solving Systems of Linear Algebraic Equations**

In the *Solving systems of linear algebraic equations* folder, you will find implementations of LU and LUP decomposition methods, as well as the computation of matrix determinants and inverses using LUP decomposition.

Implemented Features:

- **LU decomposition**
- **LUP decomposition**
- **Determinant calculation**
- **Matrix inversion using LUP decomposition**

### **Nonlinear Optimization**

 In the *Nonlinear Optimization* folder, various algorithms have been implemented for nonlinear optimization.

Implemented Algorithms:

- **Golden Section Method**
- **Coordinate Axis Search**
- **Nelder-Mead Simplex Method**
- **Hooke-Jeeves Method**

These implementations are designed to provide efficient solutions for solving nonlinear optimization problems. The algorithms aim to find optimal solutions by iteratively improving the objective function, employing diverse optimization strategies.

### Optimization with Gradient Information

Optimization with gradient information involves utilizing derivative information (such as gradients) to optimize functions efficiently. It is particularly useful for solving unconstrained optimization problems.

Implemented Methods:

- **Steepest Descent Method (Gradient Descent):**
    - Algorithm for finding the minimum of a function by moving in the direction opposite to the gradient.
- **Newton-Raphson Method:**
    - Iterative optimization algorithm that uses both first and second-order derivatives to converge to the minimum of a function.
- **Gauss-Newton Method:**
    - Optimization algorithm specifically designed for solving nonlinear least-squares problems. It iteratively refines estimates of parameters using gradient and Jacobian information.

### Constrained Optimization

Implemented Methods:

- **Box Method:**
    - The Box Method is utilized for solving optimization problems with constraints. The algorithm iteratively refines the search space by considering both explicit and implicit constraints.

- **Transformation to Unconstrained Problem - Mixed Approach:**
    - The repository features a method for transforming constrained optimization problems into unconstrained problems using a mixed approach. This approach addresses both inequality and equality constraints, providing an effective solution strategy.
