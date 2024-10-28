import math
import random
from graphviz import Digraph
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        #use numpy to add two tensors
        #other could pontentially be a scalar
        
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f"**{other}")
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out


    def __mul__(self, other):
        #element-wise multiplication
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.multiply(self.data, other.data), (self, other), "*")

        def __backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = __backward
        return out
    
    
    def __matmul__(self, other):
        #matrix multiplication

        # [[1,2],[1,2]] @ [[1,2],[1,2]] = [[3,6],[3,6]] = c

        # the derivative of the matmul with respect to c (i.e c = a @ b)
        # would be dc/da, dc/db. We need to compute how each element of a or b
        # affects c. 
        # A = [[a11m a12],[a21, a22]] B = [[b11, b12],[b21, b22]] 
        # C = [[a11*b11 + a12*b21, a11*b12 + a12*b22],[a21*b11 + a22*b21, a21*b12 + a22*b22]]
        # 
        # dc/da = dc11/da, dc12/da, dc21/da, dc22/da (partial derivatives of c 
        # with respect to a == jacobian)
        # Since this takes too much, we can use the transposed matrix, if there's a loss
        # function after C, then we want to compute dL/dA = dL/dC * dC/dA (chain rule) =
        # dL/dC * B^T (b transposed, dC/dA = B^T), * is matmul
        #
        # 

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")
        def _backward():            
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = ((np.exp(2 * x) - 1) / (np.exp(2 * x) + 1))
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        
        out = Tensor(np.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out


    def backward(self):
        self.grad = np.ones_like(self.data, dtype=np.float32)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            v._backward()

# def trace(root):
#     nodes, edges = set(), set()

#     def build(v):
#         if v not in nodes:
#             nodes.add(v)
#             for child in v._prev:
#                 edges.add((child, v))
#                 build(child)

#     build(root)
#     return nodes, edges


# def draw_dot(root, format="svg", rankdir="LR"):
#     """
#     format: png | svg | ...
#     rankdir: TB (top to bottom graph) | LR (left to right)
#     """
#     assert rankdir in ["LR", "TB"]
#     nodes, edges = trace(root)
#     dot = Digraph(
#         format=format, graph_attr={"rankdir": rankdir}
#     )  # , node_attr={'rankdir': 'TB'})

#     for n in nodes:
#         dot.node(
#             name=str(id(n)),
#             label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
#             shape="record",
#         )
#         if n._op:
#             dot.node(name=str(id(n)) + n._op, label=n._op)
#             dot.edge(str(id(n)) + n._op, str(id(n)))

#     for n1, n2 in edges:
#         dot.edge(str(id(n1)), str(id(n2)) + n2._op)

#     dot.render("graph", format="png", cleanup=True)
#     return dot


# class Neuron():
#     def __init__(self, nin, shape):
#         self.w = [Tensor(np.random.uniform(low=-1, high=1, size = shape)) for _ in range(nin)]
#         self.b = Tensor(random.uniform(-1, 1)) 
    
#     def __call__(self, x):
#         # x * w + b
        
#         assert self.w[0].data.shape == x.shape
           
#         act =  np.sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
#         # out = act.tanh()
#         # return out
    
#     def parameters(self):
#         return self.w + [self.b]

# class Layer:
#     def __init__(self, nin, nout):
#         self.neurons = [Neuron(nin) for _ in range(nout)]
    
#     def __call__(self, x):
#         outs = [neuron(x) for neuron in self.neurons]

#         return outs[0] if len(outs) == 1 else outs 
 
#     def parameters(self):
#         return [p for neuron in self.neurons for p in neuron.parameters()]
# class MLP:
#     def __init__(self, nin, nouts):
#         sz = [nin] + nouts
#         self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]   

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":

    a = Tensor(np.array([[1,2],[1,2]]))
    b = Tensor(np.array([[1,2],[1,2]]))

    c = a @ b
    
    

    
    c.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)
    # L = c.tanh()
    # L.backward()
    #c.backward()
    


    # n = MLP(3, [4,4,1])
    # print(n.parameters())
    # xs = [
    #    [2.0, 3.0, -1.0],
    #    [3.0, -1.0, 0.5],
    #    [0.5, 1.0, 1.0],
    #    [1.0, 1.0, -1.0]
    # ]
    


    # #Implement the trainning loop

    # ys = [1.0, -1.0, -1.0, 1.0]
    # for epoch in range(20):
    #     ypred = [n(x) for x in xs]
    #     loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), start=Value(0))
    #     loss.backward()
    #     for p in n.parameters():
    #         p.data += -0.01 * p.grad


    #     print(f"Epoch {epoch} Loss {loss.data}")
