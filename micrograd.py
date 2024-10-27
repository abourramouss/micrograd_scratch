import math
import random
from graphviz import Digraph
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        
        assert self.data.shape == other.data.shape
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        #use numpy to add two tensors

        out = Tensor(np.add(self.data, other.data), (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

#     def __sub__(self, other):
#         return self + (-other)

#     def __truediv__(self, other):
#         return self * other**-1

#     def __pow__(self, other):
#         assert isinstance(other, (int, float))
#         out = Value(self.data**other, (self,), f"**{other}")
#         def _backward():
#             self.grad += other * self.data**(other-1) * out.grad
#         out._backward = _backward
#         return out

    
#     def __mul__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data * other.data, (self, other), "*")

#         def _backward():
#             self.grad += other.data * out.grad
#             other.grad += self.data * out.grad

#         out._backward = _backward
#         return out

#     def __rmul__(self, other):
#         return self * other

#     def tanh(self):
#         x = self.data
#         t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

#         out = Value(t, (self,), "tanh")

#         def _backward():
#             self.grad += (1 - t**2) * out.grad

#         out._backward = _backward
#         return out
    
#     def exp(self):
#         x = self.data
        
#         out = Value(math.exp(x), (self,), "exp")

#         def _backward():
#             self.grad += out.data * out.grad

#         out._backward = _backward
#         return out


#     def backward(self):
#         self.grad = 1
#         topo = []
#         visited = set()

#         def build_topo(v):
#             if v not in visited:
#                 visited.add(v)
#                 for child in v._prev:
#                     build_topo(child)
#                 topo.append(v)

#         build_topo(self)

#         for v in reversed(topo):
#             v._backward()

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
#     def __init__(self, nin):
#         self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
#         self.b = Value(random.uniform(-1, 1)) 
    
#     def __call__(self, x):
#         # x * w + b
        
#         act =  sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
#         out = act.tanh()
#         return out
    
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

    
    arr = np.array([1,2,3])
    arr1 = np.array([1,2,3])
    
    # let's make this easier by using numpy
    a = Tensor(arr)
    b = Tensor(arr1)
    print(a + b)
    
    
    print(arr.shape == arr1.shape)

    print(arr.shape)
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
