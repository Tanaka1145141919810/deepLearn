import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.random import PRNGKey, normal

# 生成数据
def generate_data(num_samples=100):
    rng = jax.random.PRNGKey(0)
    X = jax.random.normal(rng, (num_samples, 1))  # 输入数据
    true_w = jnp.array([[2.0]])
    true_b = jnp.array([5.0])
    noise = 0.1 * jax.random.normal(rng, (num_samples, 1))
    y = jnp.dot(X, true_w) + true_b + noise
    return X, y

# 初始化权重和偏置
def init_params(key, input_size, hidden_size, output_size):
    key1, key2, key3 = jax.random.split(key, 3)
    w1 = normal(key1, (input_size, hidden_size)) * 0.1
    b1 = jnp.zeros((hidden_size,))
    w2 = normal(key2, (hidden_size, output_size)) * 0.1
    b2 = jnp.zeros((output_size,))
    return [w1, b1, w2, b2]

# 定义MLP网络
def mlp(params, X):
    w1, b1, w2, b2 = params
    hidden = jax.nn.relu(jnp.dot(X, w1) + b1)  # 隐藏层
    output = jnp.dot(hidden, w2) + b2  # 输出层
    return output

# 损失函数（均方误差）
def loss_fn(params, X, y):
    preds = mlp(params, X)
    return jnp.mean((preds - y) ** 2)

# 梯度下降优化
def update_params(params, grads, lr=0.01):
    return [p - lr * g for p, g in zip(params, grads)]

# 主函数
def train_linear_regression():
    # 超参数
    num_samples = 100
    input_size = 1
    hidden_size = 10
    output_size = 1
    num_epochs = 1000
    lr = 0.01

    # 数据生成
    X, y = generate_data(num_samples)

    # 初始化参数
    key = PRNGKey(42)
    params = init_params(key, input_size, hidden_size, output_size)

    # 编译加速
    loss_fn_jit = jit(loss_fn)
    grad_fn = jit(grad(loss_fn))

    # 训练循环
    for epoch in range(num_epochs):
        # 计算梯度
        grads = grad_fn(params, X, y)
        
        # 更新参数
        params = update_params(params, grads, lr)

        # 打印损失
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            current_loss = loss_fn_jit(params, X, y)
            print(f"Epoch {epoch + 1}, Loss: {current_loss:.4f}")

    return params

if __name__ == "__main__":
    trained_params = train_linear_regression()
        
