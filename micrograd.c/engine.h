#ifndef ENGINE_H
#define ENGINE_H

// Structure representing a node in the computational graph
typedef struct Value { 
    double data;           // The value stored in this node
    double grad;           // The gradient of this node
    void (*_backward)(struct Value* v);  // Pointer to backward function
    struct Value** _prev;  // Array of pointers to previous nodes
    int _prev_count;       // Number of previous nodes
    char* _op;             // String representing the operation
} Value;

Value* value_new(double data, Value** children, int num_children, const char* op);  // Create a new Value node
void value_free(Value* v);  // Free memory allocated for a Value node
Value* value_add(Value* a, Value* b);  // Addition operation
void value_add_backward(Value* v);  // Backward pass for addition
Value* value_mul(Value* a, Value* b);  // Multiplication operation
void value_mul_backward(Value* v);  // Backward pass for multiplication
Value* value_pow(Value* a, double power);  // Power operation
void value_pow_backward(Value* v);  // Backward pass for power operation
Value* value_relu(Value* a);  // ReLU activation function
void value_relu_backward(Value* v);  // Backward pass for ReLU
Value* value_neg(Value* a);  // Negation operation
Value* value_sub(Value* a, Value* b);  // Subtraction operation
Value* value_div(Value* a, Value* b);  // Division operation
void build_topo(Value* v, Value*** sorted, int* size, int* capacity);  // Build topological order of the computation graph
void backward(Value* v);  // Perform backward pass (backpropagation)

#endif // ENGINE_H
