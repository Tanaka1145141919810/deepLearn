from engine import Value
import numpy as np
def grad_show():
    fx_grad_arr = []
    for i in range(10):
        for j in range(10):
            x = -1 + i/5
            y = -1 + j/5
            x = Value(x)
            y = Value(y)
            v1 = x**2
            v2 = y**2
            out = v1 + v2
            out.backward()
            fx_grad = (x.data,y.data,x.grad,y.grad,out.data)
            fx_grad_arr.append(fx_grad)
    minindex ,_ = min(enumerate(fx_grad_arr) , key=lambda x : x[1][4])
    for i in fx_grad_arr:
        print(i)
    print(fx_grad_arr[minindex])
    
        
if __name__ == "__main__":
    grad_show() 