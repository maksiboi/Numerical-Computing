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
 
### Numerical Differentiation
Within the Numerical Differentiation folder, a collection of algorithms has been implemented to address the numerical challenges associated with derivative calculations. These methods play a crucial role in approximating derivatives when analytical solutions are not readily available.

Implemented Algorithms:

- **Euler Method:**
    - The Euler method offers a simple yet effective approach for numerical integration. It is employed to approximate derivatives by discretizing the function and evaluating the slope at each step.
- **Backward Euler Method:**
    - The backward Euler method, implemented here, provides an alternative approach for numerical differentiation. It involves solving equations backward in time, enhancing stability in certain scenarios.
- **Trapezoidal Rule:**
    - Leveraging the trapezoidal rule, this method provides a numerical solution for differentiation. By approximating the area under the curve, it yields an estimate of the derivative.
- **Runge-Kutta 4th Order:**
    - The Runge-Kutta 4th Order method is a higher-order numerical technique for solving differential equations. In the context of numerical differentiation, it offers increased accuracy by considering multiple points in each step.
