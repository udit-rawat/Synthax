# Source Code

## Data Initialization

JAX compartible data Initialisation for experiment , feel free to change params as your preference!

To initialize data for the problem set. Feel free to copy and use it:

```python
# Set a random seed for reproducibility
jax.random.PRNGKey(0)

# Number of data points
num_points = 1000

# Features
X = jax.random.normal(key=jax.random.PRNGKey(0), shape=(num_points, 2))

# True coefficients
true_coefficients = jnp.array([2.5, -1.0])

# True bias
true_bias = 5.0

# Generate target values with some noise
y = jnp.dot(X, true_coefficients) + true_bias + jax.random.normal(key=jax.random.PRNGKey(1), shape=(num_points, 1)) * 0.5

# Print shapes for verification
print("X shape:", X.shape)
print("y shape:", y.shape)
```

## Functions

Functions built up from scratch in JAX compartible format for the process

JAX platform optimized cost function to obtain cost after each epoch

```python
def cost(X, w, b, y):
    """
    Compute the mean squared error for linear regression.

    Parameters:
    - X: Input feature matrix.
    - w: Weight matrix.
    - b: Bias vector.
    - y: Target feature vector.

    Returns:
    - Mean squared error between predicted and actual target values.
    """
    # Predicted values using linear regression
    y_pred = jnp.dot(X, w) + b

    # Mean squared error calculation
    mse = jnp.mean((y_pred - y)**2)

    return mse

```

JAX platform optimized gradient descent optimizer

```python
def gradient_descent(X, w, b, y, learning_rate=0.01):
    """
    Perform one step of gradient descent optimization for linear regression.

    Parameters:
    - X: Input feature matrix.
    - w: Weight matrix.
    - b: Bias vector.
    - y: Target feature vector.
    - learning_rate: Step size for weight and bias updates.

    Returns:
    - Updated weight matrix (w) and bias vector (b) after one optimization step.
    """
    # Compute gradients with respect to weights and biases
    grad_w = jax.grad(cost, argnums=1)(X, w, b, y)
    grad_b = jax.grad(cost, argnums=2)(X, w, b, y)

    # Update weights and biases using the gradient and learning rate
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    return w, b

```

JAX platform optimized Newton second moment update optimizer

```python

def newton_momentum_update(X, y, w_init, b_init, alpha=0.01, beta=0.9,
                            num_iterations=1000, cost_display_interval=20,
                            patience=5, tolerance=1e-6):
    """
    Perform Newton Momentum update for linear regression.

    Parameters:
    - X: Input feature matrix.
    - y: Target feature vector.
    - w_init: Initial weight matrix. If None, it is initialized with zeros.
    - b_init: Initial bias vector. If None, it is initialized with zeros.
    - alpha: Learning rate.
    - beta: Momentum parameter.
    - num_iterations: Number of iterations.
    - cost_display_interval: Interval for displaying cost during training.
    - patience: Number of consecutive iterations with cost difference less than tolerance to trigger early stopping.
    - tolerance: Tolerance for cost difference to trigger early stopping.

    Returns:
    - Dictionary containing the weight matrix (w), bias vector (b), and cost value for each iteration.
    """
    # Example usage:
    # result_newton = newton_momentum_update(X, y, w_init=w, b_init=b)
    # Access results using result_newton['w'], result_newton['b'], result_newton['cost']

    # Initialize parameters
    w = w_init if w_init is not None else jnp.zeros((X.shape[1], 1))
    b = b_init if b_init is not None else jnp.zeros((1))

    # Initialize variables for early stopping
    consecutive_low_difference = 0

    # Initialize dictionary to store results
    optimization_results = {'w': [], 'b': [], 'cost': []}

    # Perform Newton Momentum update
    for i in range(num_iterations):
        # Compute cost
        y_pred = jnp.dot(X, w) + b
        cost = jnp.mean((y_pred - y)**2)

        # Compute gradient
        gradient_w = 2 * jnp.dot(X.T, (y_pred - y))
        gradient_b = 2 * jnp.sum(y_pred - y)

        # Compute Hessian
        hessian_w = 2 * jnp.dot(X.T, X)
        hessian_b = 2 * X.shape[0]

        # Update direction
        update_direction_w = jnp.linalg.solve(hessian_w, -gradient_w)
        update_direction_b = -gradient_b / hessian_b

        # Update with momentum
        if i == 0:
            momentum_w = jnp.zeros_like(update_direction_w)
            momentum_b = 0.0
        else:
            momentum_w = beta * momentum_w + (1 - beta) * update_direction_w
            momentum_b = beta * momentum_b + (1 - beta) * update_direction_b

        # Parameter update
        w += alpha * momentum_w
        b += alpha * momentum_b

        # Save results
        optimization_results['w'].append(w.copy())
        optimization_results['b'].append(b.copy())
        optimization_results['cost'].append(cost)

        # Print intermediate results at specified interval
        if i % cost_display_interval == 0:
            print(f"Iteration {i+1} - Newton Moment Update Cost: {cost}")

        # Early stopping check
        if i > 0:
            cost_difference = abs(previous_cost - cost)
            if cost_difference < tolerance:
                consecutive_low_difference += 1
            else:
                consecutive_low_difference = 0

            if consecutive_low_difference >= patience:
                print(f"Early stopping at iteration {i+1} due to low cost difference.")
                break

        # Save current cost for the next iteration
        previous_cost = cost

    return optimization_results


```

Main function to run the gradient descent loop

```python


def jax_optimized_gradient_loop(X, y, w_init, b_init, learning_rate=0.01,
                                 patience=5, cost_display_interval=20,
                                  tolerance=1e-6, epochs=1000):
    """
    Perform JAX platform optimized gradient descent optimization loop for linear regression.

    Parameters:
    - X: Input feature matrix.
    - y: Target feature vector.
    - w_init: Initial weight matrix.
    - b_init: Initial bias vector.
    - learning_rate: Step size for weight and bias updates.
    - patience: Number of consecutive iterations with cost difference less than tolerance to trigger early stopping.
    - cost_display_interval: Interval for displaying cost during training.
    - tolerance: Tolerance for cost difference to trigger early stopping.
    - epochs: Number of iterations for the optimization loop.

    Returns:
    - Dictionary containing the weight matrix (w), bias vector (b), and cost value for each iteration.
    """
    # Example usage:
    # result = jax_optimized_gradient_loop(X, y, w_init, b_init)
    # Access results using result['w'], result['b'], result['cost']
    # Initialize parameters
    w = w_init.copy()
    b = b_init.copy()

    # Initialize variables for early stopping
    consecutive_low_difference = 0

    # Initialize dictionary to store results
    optimization_results = {'w': [], 'b': [], 'cost': []}

    # Perform gradient descent optimization loop
    for i in range(epochs):
        # Update weights and biases using the gradient descent function
        w, b = gradient_descent(X, w, b, y, learning_rate)

        # Compute cost after each epoch
        cost_value = cost(X, w, b, y)

        # Save results
        optimization_results['w'].append(w.copy())
        optimization_results['b'].append(b.copy())
        optimization_results['cost'].append(cost_value)

        # Display cost at specified interval
        if i % cost_display_interval == 0:
            print(f"Iteration {i+1} - Gradient Descent Cost: {cost_value}")

        # Early stopping check
        if i > 0:
            cost_difference = abs(previous_cost - cost_value)
            if cost_difference < tolerance:
                consecutive_low_difference += 1
            else:
                consecutive_low_difference = 0

            if consecutive_low_difference >= patience:
                print(f"Early stopping at iteration {i+1} due to low cost difference.")
                break

        # Save current cost for the next iteration
        previous_cost = cost_value

    return optimization_results
```

## Visualization

For Visualising the change in the values with respect to both optimizer

```python
def plot_cost_comparison(dict1, dict2):
     # Create dataframes from dictionaries
df1 = pd.DataFrame({'Iteration': range(1, len(dict1['cost']) + 1)
        , 'Cost': dict1['cost'], 'Optimizer': 'Gradient Descent'})
df2 = pd.DataFrame({'Iteration': range(1, len(dict2['cost']) + 1),
         'Cost': dict2['cost'], 'Optimizer': 'Newton second moment update'})

    # Concatenate dataframes
    df = pd.concat([df1, df2])

    # Plot using Seaborn
    sns.set(style="whitegrid")
    g = sns.FacetGrid(df, col="Optimizer", height=6, aspect=1)
    g.map(plt.plot, "Iteration", "Cost", marker="o", color="b")
    g.fig.tight_layout(pad=2.0)


    # Set custom title for the facet grid
    plt.subplots_adjust(top=0.9, hspace=0.5)  # Adjust as needed
    g.fig.suptitle("Cost Comparison between Gradient descent and Newton second moment update")

    # Show the plot
    plt.show()
```
