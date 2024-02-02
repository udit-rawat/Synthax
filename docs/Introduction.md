# Introduction

This experiment focuses on the nature of optimization approaches used for deep learning. It creates the model from scratch using JAX and demonstrates the symbolic nature of equations using Sympy.

## Use Case

The main purpose of this project is to understand two variants of optimizers: gradient descent and Newton's second moment update using the Hessian matrix. To compare these, a minimum viable model is produced from scratch based on JAX, and various demonstrations of processes involved in deep learning are explored.

## Key Highlights

- Utilization of JAX for efficient numerical optimization.
- Sympy's symbolic mathematics for clear formulation and demonstration.
- Testing and comparison of simple gradient descent and Newton's second moment.
- Using visualization to demonstrate the complex nature of the optimization process.
- Explore the less popular approach of Newton's second moment update with the more popularly used gradient descent algorithm of optimization.

## Gradient Descent Optimization

1. Update Rule: In each iteration, the weights (parameters) are updated in the opposite direction of the gradient, aiming to minimize the loss function.

2. Learning Rate:The learning rate determines the step size of each update, influencing the convergence speed and stability of the algorithm.

3. Convergence: Gradient descent continues iterating until the algorithm converges to a minimum, where the gradient becomes zero.

4. Global Minima Challenge: It may get stuck in local minima, especially in non-convex loss landscapes, affecting the global optimality.

5. Computational Efficiency: Efficient for large datasets but might be computationally expensive in high-dimensional spaces.

## Newton's Second Moment Update Optimization

1. Hessian Matrix: Utilizes the Hessian matrix, which incorporates second-order information about the loss function, providing a more accurate and efficient optimization direction.

2. Update Rule: Computes the Newton update direction by solving a linear system involving the Hessian and gradient, leading to more precise parameter adjustments.

3. Learning Rate Absence: Newton's method does not require a fixed learning rate, as the update direction is determined directly from the Hessian.

4. Quicker Convergence: Often converges faster than gradient descent, especially in scenarios with strong curvatures or complex loss landscapes.

5. Challenges: Computational cost of computing and inverting the Hessian, and potential numerical instability in ill-conditioned matrices. Requires heavy dependence on parameter initialization.
