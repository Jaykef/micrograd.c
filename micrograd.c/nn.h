#ifndef NN_H
#define NN_H

#include "engine.h"

typedef struct Module {
    Value** (*parameters)(struct Module*);
    int (*parameters_count)(struct Module*);
    void (*zero_grad)(struct Module*);
} Module;

typedef struct Neuron {
    Module base;
    Value** w;
    Value* b;
    int nin;
    int nonlin;
} Neuron;

typedef struct Layer {
    Module base;
    Neuron** neurons;
    int nin;
    int nout;
} Layer;

typedef struct MLP {
    Module base;
    Layer** layers;
    int nlayers;
} MLP;

// Module functions
void module_zero_grad(Module* m);

// Neuron
Neuron* neuron_new(int nin, int nonlin);
Value* neuron_call(Neuron* n, Value** x);
Value** neuron_parameters(Module* m);
int neuron_parameters_count(Module* m);
char* neuron_repr(Neuron* n);

// Layer
Layer* layer_new(int nin, int nout, int nonlin);
Value** layer_call(Layer* l, Value** x);
Value** layer_parameters(Module* m);
int layer_parameters_count(Module* m);
char* layer_repr(Layer* l);

// MLP
MLP* mlp_new(int nin, int* nouts, int nlayers);
Value* mlp_call(MLP* m, Value** x);
Value** mlp_parameters(Module* m);
int mlp_parameters_count(Module* m);
char* mlp_repr(MLP* m);

// Free
void neuron_free(Neuron* n);
void layer_free(Layer* l);
void mlp_free(MLP* m);

#endif // NN_H
