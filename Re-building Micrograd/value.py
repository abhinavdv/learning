from graphviz import Digraph
import graphviz
import math
################ Initial Value class
# class Value:

#     def __init__(self, data):
#         self.data = data

#     def __repr__(self):
#         return f"Value(data={self.data})"
    
#     def __add__(self, other):
#         out = Value(self.data + other.data)
#         return out

#     def __mul__(self, other):
#         out = Value(self.data * other.data)
#         return out

################ Adding connective tissue. Connect node to its children when performing addition or multiplication
class Value:

    def __init__(self, data, _children = (), _op = '', label = '', grad=0.0):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad =  grad
        self._backward = lambda: None


    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        #for adding a number to a value type example: a + 1
        print("ghfghfgh ")
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
           # we have += here because when the same node is used multiple times, the gradients deposited by each parent node add up
           self.grad += out.grad
           other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
       return self + other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
           self.grad += other.data * out.grad
           other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
      assert isinstance(other, (int, float)), "only works for int/float powers for now"
      out = Value(self.data**other, (self, ), f'**{other}')
      
      def _backward():
         self.grad += other * (self.data ** (other-1)) * out.grad
      out._backward = _backward
      return out
    
    def __rmul__(self, other):
       return self * other

    def __truediv__(self, other):
       return self * other**-1
    
    def __neg__(self):
       return self * -1
    
    def __sub__(self, other):
       return self + (-other)
    
    def tanh(self):
        n = self.data
        out = Value((math.exp(2*n) - 1)/(math.exp(2*n)+1), (self, ), 'tanh')
        
        def _backward():
           self.grad += (1 - out.data**2) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        n = self.data
        out = Value(math.exp(self.data), (self, ), 'tanh')
        def _backward():
           self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
      topo = []
      visited_nodes = set()
      def build_topo(curr): 
        if curr not in visited_nodes:
            visited_nodes.add(curr)
            for child in curr._prev:
              build_topo(child)
            topo.append(curr)
      
      build_topo(self)
      print(topo)

      self.grad = 1.0
      for i in range(len(topo)-1,-1,-1):
         topo[i]._backward()
          
          

##### Just some copy pasted code to create a visualization.


def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot


def run_neuron_with_tanh():
   #building a neuron -> L = tanh(w1x1 + w2x2 +b)
    # inputs x1, x2
    x1 = Value(2.0, label = 'x1')
    x2 = Value(0.0, label = 'x2')
    
    # weights w1, w2
    w1 = Value(-3.0, label = 'w1')
    w2 = Value(1.0, label= 'w2')

    # bias of neuron
    b = Value(6.8813735870195432, label='b')

    # w1x1 + w2x2 + b
    w1x1 = w1 * x1 ; w1x1.label = 'w1x1'
    w2x2 = w2 * x2 ; w2x2.label = 'w2x2'
    w1x1w2x2 = w1x1 + w2x2; w1x1w2x2.label = 'w1x1w2x2'
    n = w1x1w2x2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    
    #manual back pass
    # o.grad = 1.0
    # o._backward()
    # n._backward()
    # w1x1w2x2._backward()
    # b._backward()
    # w1x1._backward()
    # w2x2._backward()
    
    #using function for entire backward 
    o.backward()
    return o

def run_neuron_without_tanh():
   #building a neuron -> L = tanh(w1x1 + w2x2 +b)
    # inputs x1, x2
    x1 = Value(2.0, label = 'x1')
    x2 = Value(0.0, label = 'x2')
    
    # weights w1, w2
    w1 = Value(-3.0, label = 'w1')
    w2 = Value(1.0, label= 'w2')

    # bias of neuron
    b = Value(6.8813735870195432, label='b')

    # w1x1 + w2x2 + b
    w1x1 = w1 * x1 ; w1x1.label = 'w1x1'
    w2x2 = w2 * x2 ; w2x2.label = 'w2x2'
    w1x1w2x2 = w1x1 + w2x2; w1x1w2x2.label = 'w1x1w2x2'
    n = w1x1w2x2 + b; n.label = 'n'
        
    # ----
    e = (2*n).exp()
    o = (e - 1) / (e + 1)
    # ----
    o.label = 'o'
    o.backward()
    return o

def draw_graph(o, filename):
    dot = draw_dot(o)
    dot.render(filename=filename, format='png', cleanup=True)

def using_pytorch():
  import torch

  x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
  x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
  w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
  w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
  b = torch.Tensor([6.8813735870195432]).double()  ; b.requires_grad = True
  n = x1*w1 + x2*w2 + b
  o = torch.tanh(n)

  print(o.data.item())
  o.backward()

  print('---')
  print('x2', x2.grad.item())
  print('w2', w2.grad.item())
  print('x1', x1.grad.item())
  print('w1', w1.grad.item())



if __name__ == "__main__":
   o1 = run_neuron_with_tanh()
   draw_graph(o1, 'o1')
   o2 = run_neuron_without_tanh()
   draw_graph(o2, 'o2')
   o3 = using_pytorch()

