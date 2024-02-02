# Process

## Introduction to process

The whole experiment is divided into three test which is based on three different approaches to explore the experiment

- Normal Init - Normal Initialization of parameter
- Better Init - Better Initialization of parameter
- Worst Init - Poor Initialization of paramater

The main crux of the exploration behind this experiment is that newton second moment update heavily influence by the initial parameter initialization , the whole experiment is evaluated through:

- The convergence precision
- Number of iteration it took
- The performance of the optimiser in various situation

## Result for each iteration

### Normal Init

```python
#Normal Initialisation
jax.random.PRNGKey(0)
w_init = jax.random.normal(key=jax.random.PRNGKey(2), shape=(X.shape[1], 1))
b_init = jax.random.normal(key=jax.random.PRNGKey(3), shape=(1,))
```

Gradient Descent

```python
# Run JAX Optimized Gradient Loop
start_time = time.time()
result_gd1 = jax_optimized_gradient_loop(X, y, w_init=w_init, b_init=b_init, learning_rate=0.01, patience=2,tolerance=0.001)
elapsed_time = time.time()-start_time
print(f"Time taken: {elapsed_time} seconds")
```

Newton second moment update

```python
# Run Newton Momentum Update
start_time = time.time()
result_nm1 = newton_momentum_update(X, y,w_init=w_init,b_init=b_init, alpha=0.01, beta=0.9, num_iterations=1000,patience=2,tolerance=0.001)
elapsed_time1 = time.time()-start_time
print(f"Time taken: {elapsed_time} seconds")
```

plot_cost_comparison(result_gd1,result_nm1)

### Better Init

```python
#Better initialization of parameters.
# -features and hyperparams remain same.
w_init = jnp.zeros((X.shape[1], 1))
b_init = jnp.zeros((1,))
```

````python
# Run JAX Optimized Gradient Loop
start_time = time.time()
result_gd2 = jax_optimized_gradient_loop(X, y, w_init, b_init, learning_rate=0.01, patience=2,tolerance=0.001)
elapsed_time2 = time.time()-start_time
print(f"Time taken: {elapsed_time} seconds")

```python
# Run Newton Momentum Update
start_time = time.time()
result_nm2 = newton_momentum_update(X, y,w_init=w_init,b_init=b_init, alpha=0.01, beta=0.9, num_iterations=1000,patience=2,tolerance=0.001)
elapsed_time3 = time.time()-start_time
print(f"Time taken: {elapsed_time} seconds")
````

```python
plot_cost_comparison(result_gd2,result_nm2)
```

### Worst Init

```python
#Worst initialization of parameters.
# -features and hyperparams remain same.
w_init = jax.random.normal(key=jax.random.PRNGKey(4), shape=(X.shape[1], 1))
b_init = jnp.array([160.0])
```

```python
# Run JAX Optimized Gradient Loop
start_time = time.time()
result_gd3 = jax_optimized_gradient_loop(X, y, w_init, b_init, learning_rate=0.01, patience=2,tolerance=0.001)
elapsed_time4 = time.time()-start_time
print(f"Time taken: {elapsed_time} seconds")
```

```python
# Run Newton Momentum Update
start_time = time.time()
result_nm3 = newton_momentum_update(X, y,w_init,b_init, alpha=0.01, beta=0.9, num_iterations=1000,patience=2,tolerance=0.001)
elapsed_time5 = time.time()-start_time
print(f"Time taken: {elapsed_time} seconds")
```

```python
plot_cost_comparison(result_gd3,result_nm3)
```
