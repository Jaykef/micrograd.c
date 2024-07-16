# micrograd.c
Port of Karpathy's <a href="https://github.com/karpathy/micrograd">migrograd</a> in pure C. Migrograd is a tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API.

### Quick Start

```bash
cd micrograd.c
make
./main
./train
```

### Example Usage

```c
#include <stdio.h>
#include <math.h>
#include "../micrograd.c/nn.c"

Value* a = value_new(-4.0);
Value* b = value_new(2.0);
Value* c = value_add(a, b);
Value* d = value_add(value_mul(a, b), value_pow(b, 3));
c = value_add(c, value_add(c, value_new(1)));
c = value_add(c, value_add(value_add(value_new(1), c), value_neg(a)));
d = value_add(d, value_add(value_mul(d, value_new(2)), value_relu(value_add(b, a))));
d = value_add(d, value_add(value_mul(value_new(3), d), value_relu(value_sub(b, a))));
Value* e = value_sub(c, d);
Value* f = value_pow(e, 2);
Value* g = value_div(f, value_new(2.0));
g = value_add(g, value_div(value_new(10.0), f));
backward(g);

double tol = 1e-4; 
printf("g->data: %.6f\n", g->data);

backward(g);

printf("a->grad: %.6f\n", a->grad);
printf("b->grad: %.6f\n", b->grad);
```

## Train a neural net
[train.c](https://github.com/Jaykef/micrograd.c/blob/main/train.c) has logic for training a simple multi-layer perceptron on 100 samples of make_moons dataset.

```bash
cd micrograd.c
make
./train
```

## Test

```bash
cd micrograd.c
make
./test
```
