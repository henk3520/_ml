import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad  # d(exp(x))/dx = exp(x)
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        sigmoid_x = 1 / (1 + math.exp(-x))
        out = Value(sigmoid_x, (self,), 'sigmoid')

        def _backward():
            self.grad += sigmoid_x * (1 - sigmoid_x) * out.grad  # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# 測試程式碼
if __name__ == "__main__":
    # 測試 exp
    x = Value(2.0, label='x')
    y = x.exp()
    y.backward()
    print(f"exp({x.data}) = {y.data}, grad of x = {x.grad}")  # 應為 e^2, grad = e^2

    # 測試 sigmoid
    x = Value(0.0, label='x')
    z = x.sigmoid()
    z.backward()
    print(f"sigmoid({x.data}) = {z.data}, grad of x = {x.grad}")  # 應為 0.5, grad = 0.25
