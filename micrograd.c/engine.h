#ifndef ENGINE_H
#define ENGINE_H

typedef struct Value { 
    // Stores a single scalar value and its gradient 
    double data;
    double grad;
    void (*_backward)(struct Value* v);
    struct Value** _prev;
    int _prev_count;
    char* _op;
} Value;

Value* value_new(double data, Value** children, int num_children, const char* op);
void value_free(Value* v);
Value* value_add(Value* a, Value* b);
void value_add_backward(Value* v);
Value* value_mul(Value* a, Value* b);
void value_mul_backward(Value* v);
Value* value_pow(Value* a, double power);
void value_pow_backward(Value* v);
Value* value_relu(Value* a);
void value_relu_backward(Value* v);
Value* value_neg(Value* a);
Value* value_sub(Value* a, Value* b);
Value* value_div(Value* a, Value* b);
void build_topo(Value* v, Value*** sorted, int* size, int* capacity);
void backward(Value* v);

#endif // ENGINE_H
