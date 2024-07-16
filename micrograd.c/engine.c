#include "engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

Value* value_new(double data) {
    Value* v = malloc(sizeof(Value));
    if (v == NULL) {
        fprintf(stderr, "Failed to allocate memory for Value\n");
        return NULL;
    }
    v->data = data;
    v->grad = 0.0;
    v->backward = NULL; 
    v->prev = NULL;
    v->prev_count = 0;
    v->op = NULL;
    return v;
}

void value_free(Value* v) {
    free(v->prev);
    free(v->op);
    free(v);
}

Value* value_add(Value* a, Value* b) {
    Value* out = value_new(a->data + b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->prev_count = 2;
    out->op = strdup("+");
    out->backward = value_add_backward;
    return out;
}

void value_add_backward(Value* v) {
    v->prev[0]->grad += v->grad;
    v->prev[1]->grad += v->grad;
}


Value* value_mul(Value* a, Value* b) {
    Value* out = value_new(a->data * b->data);
    out->prev = malloc(2 * sizeof(Value*));
    out->prev[0] = a;
    out->prev[1] = b;
    out->prev_count = 2;
    out->op = strdup("*");
    out->backward = value_mul_backward;
    return out;
}

void value_mul_backward(Value* v) {
    v->prev[0]->grad += v->prev[1]->data * v->grad;
    v->prev[1]->grad += v->prev[0]->data * v->grad;
}

Value* value_pow(Value* a, double power) {
    Value* out = value_new(pow(a->data, power));
    out->prev = malloc(sizeof(Value*));
    out->prev[0] = a;
    out->prev_count = 1;
    char power_str[20];
    snprintf(power_str, sizeof(power_str), "**%.2f", power);
    out->op = strdup(power_str);
    out->backward = value_pow_backward;
    return out;
}

void value_pow_backward(Value* v) {
    double power;
    sscanf(v->op + 2, "%lf", &power);
    v->prev[0]->grad += power * pow(v->prev[0]->data, power - 1) * v->grad;
}

Value* value_relu(Value* a) {
    Value* out = value_new(a->data > 0 ? a->data : 0);
    out->prev = malloc(sizeof(Value*));
    out->prev[0] = a;
    out->prev_count = 1;
    out->op = strdup("ReLU");
    out->backward = value_relu_backward;
    return out;
}

void value_relu_backward(Value* v) {
    v->prev[0]->grad += (v->prev[0]->data > 0) ? v->grad : 0;
}

Value* value_neg(Value* a) {
    return value_mul(a, value_new(-1));
}

Value* value_sub(Value* a, Value* b) {
    return value_add(a, value_neg(b));
}

Value* value_div(Value* a, Value* b) {
    return value_mul(a, value_pow(b, -1));
}

void build_topo(Value* v, Value*** sorted, int* size, int* capacity) {
    
    for (int i = 0; i < v->prev_count; i++) {
        if (v->prev[i]->backward) {
            build_topo(v->prev[i], sorted, size, capacity);
        }
    }
    
    for (int i = 0; i < *size; i++) {
        if ((*sorted)[i] == v) return;
    }
    
    if (*size >= *capacity) {
        *capacity *= 2;
        *sorted = realloc(*sorted, *capacity * sizeof(Value*));
    }
    
    (*sorted)[(*size)++] = v;
}

void backward(Value* v) {
    int size = 0;
    int capacity = 10;
    Value** topo = malloc(capacity * sizeof(Value*));
    build_topo(v, &topo, &size, &capacity);

    v->grad = 1;
    for (int i = size - 1; i >= 0; i--) {
        if (topo[i]->backward) {
            topo[i]->backward(topo[i]);
        }
    }

    free(topo);
}

