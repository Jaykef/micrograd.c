#include "engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Creates a new Value node
Value* value_new(double data, Value** children, int num_children, const char* op) {
    Value* v = malloc(sizeof(Value)); // Allocates memory for new Value
    if (v == NULL) {
        fprintf(stderr, "Failed to allocate memory for Value\n");
        return NULL;
    }
    v->data = data;
    v->grad = 0.0;
    v->_backward = NULL;
    v->_prev = malloc(num_children * sizeof(Value*));
    memcpy(v->_prev, children, num_children * sizeof(Value*)); // Copies children to previous nodes
    v->_prev_count = num_children;
    v->_op = strdup(op);
    return v;
}

// Frees memory allocated for a Value node
void value_free(Value* v) {
    free(v->_prev);
    free(v->_op);
    free(v);
}

// Addition operation
Value* value_add(Value* a, Value* b) {
    Value* children[] = {a, b};
    Value* out = value_new(a->data + b->data, children, 2, "+");

    out->_backward = value_add_backward;
    return out;
}

// Backward pass for addition
void value_add_backward(Value* v) {
    v->_prev[0]->grad += v->grad;
    v->_prev[1]->grad += v->grad;
}

// Multiplication operation
Value* value_mul(Value* a, Value* b) {
    Value* children[] = {a, b};
    Value* out = value_new(a->data * b->data, children, 2, "*");

    out->_backward = value_mul_backward;
    return out;
}

// Backward pass for multiplication
void value_mul_backward(Value* v) {
    v->_prev[0]->grad += v->_prev[1]->data * v->grad;
    v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

// Power operation
Value* value_pow(Value* a, double power) {
    Value* children[] = {a};
    char power_str[20];
    snprintf(power_str, sizeof(power_str), "**%.2f", power);
    Value* out = value_new(pow(a->data, power), children, 1, power_str);

    out->_backward = value_pow_backward;
    return out;
}

// Backward pass for power operation
void value_pow_backward(Value* v) {
    double power;
    sscanf(v->_op + 2, "%lf", &power);
    v->_prev[0]->grad += power * pow(v->_prev[0]->data, power - 1) * v->grad;
}

// ReLU activation function
Value* value_relu(Value* a) {
    Value* children[] = {a};
    Value* out = value_new(a->data > 0 ? a->data : 0, children, 1, "ReLU");

    out->_backward = value_relu_backward;
    return out;
}

// Backward pass for ReLU
void value_relu_backward(Value* v) {
    v->_prev[0]->grad += (v->_prev[0]->data > 0) ? v->grad : 0;
}

// Negation operation
Value* value_neg(Value* a) {
    return value_mul(a, value_new(-1, NULL, 0, ""));
}

// Subtraction operation
Value* value_sub(Value* a, Value* b) {
    return value_add(a, value_neg(b));
}

// Division operation
Value* value_div(Value* a, Value* b) {
    return value_mul(a, value_pow(b, -1));
}

// Builds topological order of the computation graph
void build_topo(Value* v, Value*** sorted, int* size, int* capacity) {
    // Recursively process children
    for (int i = 0; i < v->_prev_count; i++) {
        if (v->_prev[i]->_backward) {
            build_topo(v->_prev[i], sorted, size, capacity);
        }
    }
    
    // Check if node is already in sorted list
    for (int i = 0; i < *size; i++) {
        if ((*sorted)[i] == v) return;
    }
    
    // Expand capacity if needed
    if (*size >= *capacity) {
        *capacity *= 2;
        *sorted = realloc(*sorted, *capacity * sizeof(Value*));
    }
    
    // Add node to sorted list
    (*sorted)[(*size)++] = v;
}

// Performs backward pass (backpropagation)
void backward(Value* v) {
    int size = 0;
    int capacity = 10;
    Value** topo = malloc(capacity * sizeof(Value*));
    build_topo(v, &topo, &size, &capacity);

    // Set gradient of output to 1
    v->grad = 1;
    
    // Perform backward pass in reverse topological order
    for (int i = size - 1; i >= 0; i--) {
        if (topo[i]->_backward) {
            topo[i]->_backward(topo[i]);
        }
    }

    free(topo);
}
