#include "nn.h"
#include "engine.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Forward declarations
Value* neuron_call(Neuron* n, Value** x);
Value** layer_call(Layer* l, Value** x);

// Sets gradients of all parameters to zero
void module_zero_grad(Module* m) {
    Value** params = m->parameters(m);
    int count = m->parameters_count(m);
    for (int i = 0; i < count; i++) {
        params[i]->grad = 0;
    }
    free(params);
}

// Creates a new neuron
Neuron* neuron_new(int nin, int nonlin) {
    Neuron* n = malloc(sizeof(Neuron));
    n->w = malloc(nin * sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        // Initialize weights with random values between -1 and 1
        n->w[i] = value_new((double)rand() / RAND_MAX * 2 - 1, NULL, 0, "");
    }
    n->b = value_new(0, NULL, 0, "");  // Initialize bias to 0
    n->nin = nin;
    n->nonlin = nonlin;
    n->base.parameters = neuron_parameters;
    n->base.parameters_count = neuron_parameters_count;
    n->base.zero_grad = module_zero_grad;
    return n;
}

// Computes the output of a neuron
Value* neuron_call(Neuron* n, Value** x) {
    Value* act = n->b;  // Start with the bias
    for (int i = 0; i < n->nin; i++) {
        // Multiply each input by its weight and add to the activation
        act = value_add(act, value_mul(n->w[i], x[i]));
    }
    return n->nonlin ? value_relu(act) : act;  // Apply ReLU if nonlinear
}

// Gets parameters of a neuron
Value** neuron_parameters(Module* m) {
    Neuron* n = (Neuron*)m;
    Value** params = malloc((n->nin + 1) * sizeof(Value*));
    memcpy(params, n->w, n->nin * sizeof(Value*));  // Copy weights
    params[n->nin] = n->b;  // Add bias as the last parameter
    return params;
}

// Get number of parameters in a neuron
int neuron_parameters_count(Module* m) {
    Neuron* n = (Neuron*)m;
    return n->nin + 1;  // Number of inputs plus bias
}

// Get string representation of a neuron
char* neuron_repr(Neuron* n) {
    char* repr = malloc(50 * sizeof(char));
    snprintf(repr, 50, "%sNeuron(%d)", n->nonlin ? "ReLU" : "Linear", n->nin);
    return repr;
}

// Creates a new layer
Layer* layer_new(int nin, int nout, int nonlin) {
    Layer* l = malloc(sizeof(Layer));
    l->neurons = malloc(nout * sizeof(Neuron*));
    for (int i = 0; i < nout; i++) {
        l->neurons[i] = neuron_new(nin, nonlin);
    }
    l->nin = nin;
    l->nout = nout;
    l->base.parameters = layer_parameters;
    l->base.parameters_count = layer_parameters_count;
    l->base.zero_grad = module_zero_grad;
    return l;
}

// Compute the output of a layer
Value** layer_call(Layer* l, Value** x) {
    Value** out = malloc(l->nout * sizeof(Value*));
    for (int i = 0; i < l->nout; i++) {
        out[i] = neuron_call(l->neurons[i], x);
    }
    return out;
}

// Gets parameters of a layer
Value** layer_parameters(Module* m) {
    Layer* l = (Layer*)m;
    int count = layer_parameters_count(m);
    Value** params = malloc(count * sizeof(Value*));
    int index = 0;
    for (int i = 0; i < l->nout; i++) {
        Value** neuron_params = neuron_parameters((Module*)l->neurons[i]);
        int neuron_param_count = neuron_parameters_count((Module*)l->neurons[i]);
        // Copy parameters of each neuron into the layer's parameter array
        memcpy(params + index, neuron_params, neuron_param_count * sizeof(Value*));
        index += neuron_param_count;
        free(neuron_params);
    }
    return params;
}

// Gets number of parameters in a layer
int layer_parameters_count(Module* m) {
    Layer* l = (Layer*)m;
    int count = 0;
    for (int i = 0; i < l->nout; i++) {
        count += neuron_parameters_count((Module*)l->neurons[i]);
    }
    return count;
}

// Gets string representation of a layer
char* layer_repr(Layer* l) {
    char* repr = malloc(200 * sizeof(char));
    char* temp = malloc(50 * sizeof(char));
    strcpy(repr, "Layer of [");
    for (int i = 0; i < l->nout; i++) {
        char* neuron_str = neuron_repr(l->neurons[i]);
        strcat(repr, neuron_str);
        if (i < l->nout - 1) strcat(repr, ", ");
        free(neuron_str);
    }
    strcat(repr, "]");
    free(temp);
    return repr;
}

// Creates a new MLP (Multi-Layer Perceptron)
MLP* mlp_new(int nin, int* nouts, int nlayers) {
    MLP* m = malloc(sizeof(MLP));
    m->layers = malloc(nlayers * sizeof(Layer*));
    m->nlayers = nlayers;
    
    int prev_nout = nin;
    for (int i = 0; i < nlayers; i++) {
        // Creates layers, making all but the last layer non-linear
        m->layers[i] = layer_new(prev_nout, nouts[i], i != nlayers - 1);
        prev_nout = nouts[i];
    }
    
    m->base.parameters = mlp_parameters;
    m->base.parameters_count = mlp_parameters_count;
    m->base.zero_grad = module_zero_grad;
    return m;
}

// Compute the output of an MLP
Value* mlp_call(MLP* m, Value** x) {
    Value** out = x;
    for (int i = 0; i < m->nlayers; i++) {
        out = layer_call(m->layers[i], out);
    }
    return out[0];  // Return the first (and only) output of the last layer
}

// Gets parameters of an MLP
Value** mlp_parameters(Module* mod) {
    MLP* m = (MLP*)mod;
    int count = mlp_parameters_count(mod);
    Value** params = malloc(count * sizeof(Value*));
    int index = 0;
    for (int i = 0; i < m->nlayers; i++) {
        Value** layer_params = layer_parameters((Module*)m->layers[i]);
        int layer_param_count = layer_parameters_count((Module*)m->layers[i]);
        // Copy parameters of each layer into the MLP's parameter array
        memcpy(params + index, layer_params, layer_param_count * sizeof(Value*));
        index += layer_param_count;
        free(layer_params);
    }
    return params;
}

// Gets number of parameters in an MLP
int mlp_parameters_count(Module* mod) {
    MLP* m = (MLP*)mod;
    int count = 0;
    for (int i = 0; i < m->nlayers; i++) {
        count += layer_parameters_count((Module*)m->layers[i]);
    }
    return count;
}

// Get string representation of an MLP
char* mlp_repr(MLP* m) {
    char* repr = malloc(500 * sizeof(char));
    strcpy(repr, "MLP of [");
    for (int i = 0; i < m->nlayers; i++) {
        char* layer_str = layer_repr(m->layers[i]);
        strcat(repr, layer_str);
        if (i < m->nlayers - 1) strcat(repr, ", ");
        free(layer_str);
    }
    strcat(repr, "]");
    return repr;
}

// Frees memory allocated for a neuron
void neuron_free(Neuron* n) {
    for (int i = 0; i < n->nin; i++) {
        value_free(n->w[i]);
    }
    free(n->w);
    value_free(n->b);
    free(n);
}

// Frees memory allocated for a layer
void layer_free(Layer* l) {
    for (int i = 0; i < l->nout; i++) {
        neuron_free(l->neurons[i]);
    }
    free(l->neurons);
    free(l);
}

// Free memory allocated for an MLP
void mlp_free(MLP* m) {
    for (int i = 0; i < m->nlayers; i++) {
        layer_free(m->layers[i]);
    }
    free(m->layers);
    free(m);
}
