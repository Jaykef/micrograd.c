#include <stdlib.h>
#include <time.h>
#include "engine.c"

typedef struct {
    Value** w;
    Value* b;
    int nin;
    int nonlin;
} Neuron;

typedef struct {
    Neuron** neurons;
    int nin;
    int nout;
} Layer;

typedef struct {
    Layer** layers;
    int nlayers;
} MLP;

Neuron* neuron_new(int nin, int nonlin) {
    Neuron* n = malloc(sizeof(Neuron));
    n->w = malloc(nin * sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        n->w[i] = value_new((double)rand() / RAND_MAX * 2 - 1);
    }
    n->b = value_new(0);
    n->nin = nin;
    n->nonlin = nonlin;
    return n;
}

Value* neuron_call(Neuron* n, Value** x) {
    Value* act = n->b;
    for (int i = 0; i < n->nin; i++) {
        act = value_add(act, value_mul(n->w[i], x[i]));
    }
    return n->nonlin ? value_relu(act) : act;
}

Layer* layer_new(int nin, int nout) {
    Layer* l = malloc(sizeof(Layer));
    l->neurons = malloc(nout * sizeof(Neuron*));
    for (int i = 0; i < nout; i++) {
        l->neurons[i] = neuron_new(nin, 1);
    }
    l->nin = nin;
    l->nout = nout;
    return l;
}

Value** layer_call(Layer* l, Value** x) {
    Value** out = malloc(l->nout * sizeof(Value*));
    if (out == NULL) {
        printf("Error: Failed to allocate memory for layer output\n");
        return NULL;
    }
    for (int i = 0; i < l->nout; i++) {
        out[i] = neuron_call(l->neurons[i], x);
        if (out[i] == NULL) {
            printf("Error: neuron_call returned NULL for neuron %d\n", i);
            free(out);
            return NULL;
        }
    }
    return out;
}

MLP* mlp_new(int nin, int* nouts, int nlayers) {
    MLP* m = malloc(sizeof(MLP));
    m->layers = malloc(nlayers * sizeof(Layer*));
    m->nlayers = nlayers;
    
    int prev_nout = nin;
    for (int i = 0; i < nlayers; i++) {
        m->layers[i] = layer_new(prev_nout, nouts[i]);
        prev_nout = nouts[i];
    }
    
    return m;
}

Value* mlp_call(MLP* m, Value** x) {
    Value** out = x;
    for (int i = 0; i < m->nlayers; i++) {
        out = layer_call(m->layers[i], out);
        if (out == NULL) {
            printf("Error: layer_call returned NULL for layer %d\n", i);
            return NULL;
        }
    }
    return out[0]; 
}

void neuron_free(Neuron* n) {
    if (n) {
        for (int i = 0; i < n->nin; i++) {
            value_free(n->w[i]);
        }
        free(n->w);
        value_free(n->b);
        free(n);
    }
}

void layer_free(Layer* l) {
    if (l) {
        for (int i = 0; i < l->nout; i++) {
            neuron_free(l->neurons[i]);
        }
        free(l->neurons);
        free(l);
    }
}

void mlp_free(MLP* m) {
    if (m) {
        for (int i = 0; i < m->nlayers; i++) {
            layer_free(m->layers[i]);
        }
        free(m->layers);
        free(m);
    }
}