#ifndef NN_H
#define NN_H
#include "engine.h"

typedef struct Neuron {
    Value** w;
    Value* b;
    int nin;
    int nonlin;
} Neuron;

typedef struct Layer {
    Neuron** neurons;
    int nin;
    int nout;
} Layer;

typedef struct MLP {
    Layer** layers;
    int nlayers;
} MLP;

Neuron* neuron_new(int nin, int nonlin);
Layer* layer_new(int nin, int nout);
MLP* mlp_new(int nin, int* nouts, int nlayers);
Value* mlp_call(MLP* m, Value** x);
Value** mlp_parameters(MLP* m);

void neuron_free(Neuron* n);
void layer_free(Layer* l);
void mlp_free(MLP* m);

int mlp_parameters_count(MLP* m);
void mlp_zero_grad(MLP* m);
void mlp_update(MLP* m, double learning_rate);

#endif // NN_H
