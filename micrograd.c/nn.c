#include "nn.h"
#include "engine.h"
#include <stdlib.h>
#include <string.h>

Value** layer_call(Layer* l, Value** x);
Value* neuron_call(Neuron* n, Value** x);

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
    }
    return out[0];
}

Value** layer_call(Layer* l, Value** x) {
    Value** out = malloc(l->nout * sizeof(Value*));
    for (int i = 0; i < l->nout; i++) {
        out[i] = neuron_call(l->neurons[i], x);
    }
    return out;
}

Value* neuron_call(Neuron* n, Value** x) {
    Value* act = n->b;
    for (int i = 0; i < n->nin; i++) {
        act = value_add(act, value_mul(n->w[i], x[i]));
    }
    return n->nonlin? value_relu(act) : act;
}

void neuron_free(Neuron* n) {
    for (int i = 0; i < n->nin; i++) {
        value_free(n->w[i]);
    }
    free(n->w);
    value_free(n->b);
    free(n);
}

void layer_free(Layer* l) {
    for (int i = 0; i < l->nout; i++) {
        neuron_free(l->neurons[i]);
    }
    free(l->neurons);
    free(l);
}

void mlp_free(MLP* m) {
    for (int i = 0; i < m->nlayers; i++) {
        layer_free(m->layers[i]);
    }
    free(m->layers);
    free(m);
}

int mlp_parameters_count(MLP* m) {
    int count = 0;
    for (int i = 0; i < m->nlayers; i++) {
        for (int j = 0; j < m->layers[i]->nout; j++) {
            count += m->layers[i]->neurons[j]->nin + 1;  // weights + bias
        }
    }
    return count;
}

Value** mlp_parameters(MLP* m) {
    int count = mlp_parameters_count(m);
    Value** params = malloc(count * sizeof(Value*));
    int index = 0;
    for (int i = 0; i < m->nlayers; i++) {
        for (int j = 0; j < m->layers[i]->nout; j++) {
            Neuron* n = m->layers[i]->neurons[j];
            for (int k = 0; k < n->nin; k++) {
                params[index++] = n->w[k];
            }
            params[index++] = n->b;
        }
    }
    return params;
}

void mlp_zero_grad(MLP* m) {
    for (int i = 0; i < m->nlayers; i++) {
        for (int j = 0; j < m->layers[i]->nout; j++) {
            Neuron* n = m->layers[i]->neurons[j];
            for (int k = 0; k < n->nin; k++) {
                n->w[k]->grad = 0;
            }
            n->b->grad = 0;
        }
    }
}

void mlp_update(MLP* m, double learning_rate) {
    for (int i = 0; i < m->nlayers; i++) {
        for (int j = 0; j < m->layers[i]->nout; j++) {
            Neuron* n = m->layers[i]->neurons[j];
            for (int k = 0; k < n->nin; k++) {
                n->w[k]->data -= learning_rate * n->w[k]->grad;
            }
            n->b->data -= learning_rate * n->b->grad;
        }
    }
}

