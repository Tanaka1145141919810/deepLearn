import numpy as np
from engine import Value

def SGD_test(lr : float = 0.3):
    x1 = Value(1)
    x2 = Value(1)
    for _ in range(10):
        v1 = x1 ** 2
        v2 = x2 ** 2
        out = v1 + v2
        out.backward()
        x1.data = x1.data - lr * x1.grad
        x2.data = x2.data - lr * x2.grad
        ## 重要 防止梯度累计
        x1.grad = 0
        x2.grad = 0
        print(x1.data)
        print(x2.data)
        
if __name__ == "__main__":
    SGD_test()