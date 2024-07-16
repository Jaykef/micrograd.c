#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "../micrograd.c/nn.h"
#include "../micrograd.c/engine.h"
#include "test.h"

void test_sanity_check(void) {
    Value* x = value_new(-4.0);
    Value* z = value_add(value_add(value_mul(value_new(2), x), value_new(2)), x);
    Value* q = value_add(value_relu(z), value_mul(z, x));
    Value* h = value_relu(value_mul(z, z));
    Value* y = value_add(value_add(h, q), value_mul(q, x));
    backward(y);

    // forward pass check
    assert(fabs(y->data - (-20)) < 1e-6);
    printf("y->data: %.1f, expected: -20.0\n", y->data);

    // backward pass check
    assert(fabs(x->grad - 46) < 1e-6);
    printf("x->grad: %.1f, expected: 46.0\n", x->grad);

    printf("test_sanity_check passed\n");
}

void test_more_ops(void) {
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
    printf("g->data: %.6f, expected: 24.704100\n", g->data);
    assert(fabs(g->data - 24.7041) < tol);
    // backward pass check
    printf("a->grad: %.6f, expected: 138.833800\n", a->grad);
    assert(fabs(a->grad - 138.8338) < tol);
    printf("b->grad: %.6f, expected: 645.577300\n", b->grad);
    assert(fabs(b->grad - 645.5773) < tol);

    printf("test_more_ops passed\n");
}
