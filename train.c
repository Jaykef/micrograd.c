#include "micrograd.c/nn.h"
#include "micrograd.c/engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BATCH_SIZE 100
#define INPUT_SIZE 2
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 1
#define NUM_LAYERS 3

void print_value(Value* v) {
    if (v) {
        printf("Value: data=%f, grad=%f\n", v->data, v->grad);
    } else {
        printf("Value is NULL\n");
    }
}

void print_model_params(MLP* model) {
    for (int i = 0; i < model->nlayers; i++) {
        for (int j = 0; j < model->layers[i]->nout; j++) {
            Neuron* neuron = model->layers[i]->neurons[j];
            printf("Layer %d, Neuron %d:\n", i, j);
            for (int k = 0; k < neuron->nin; k++) {
                printf("  w[%d]: data=%f, grad=%f\n", k, neuron->w[k]->data, neuron->w[k]->grad);
            }
            printf("  b: data=%f, grad=%f\n", neuron->b->data, neuron->b->grad);
        }
    }
}

int main(void) {
    // Load data from make_moons.csv
    double X[BATCH_SIZE][INPUT_SIZE];
    int y[BATCH_SIZE];
    FILE *fp = fopen("data/make_moons.csv", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }
    
    char buffer[1024];
    fgets(buffer, sizeof(buffer), fp);
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (fscanf(fp, "%lf,%lf,%d", &X[i][0], &X[i][1], &y[i]) != 3) {
            fprintf(stderr, "Error reading line %d from file\n", i+1);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    printf("Data loaded successfully\n");

    // Initialize model
    int nouts[NUM_LAYERS - 1] = {HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    MLP* model = mlp_new(INPUT_SIZE, nouts, NUM_LAYERS - 1);
    if (model == NULL) {
        fprintf(stderr, "Failed to create MLP\n");
        return 1;
    }

    printf("Model initialized\n");

    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        // Define loss function
        Value* loss_value = value_new(0);
        for (int i = 0; i < BATCH_SIZE; i++) {
            Value** inputs = malloc(INPUT_SIZE * sizeof(Value*));
            if (inputs == NULL) {
                fprintf(stderr, "Memory allocation failed for inputs\n");
                mlp_free(model);
                value_free(loss_value);
                return 1;
            }
            for (int j = 0; j < INPUT_SIZE; j++) {
                inputs[j] = value_new(X[i][j]);
            }
            Value* output = mlp_call(model, inputs);
            if (output == NULL) {
                fprintf(stderr, "mlp_call returned NULL\n");
                mlp_free(model);
                value_free(loss_value);
                for (int j = 0; j < INPUT_SIZE; j++) {
                    value_free(inputs[j]);
                }
                free(inputs);
                return 1;
            }
            Value* label = value_new(y[i] == 0 ? -1 : 1);
            Value* score = value_mul(output, label);
            Value* loss = value_add(value_relu(value_add(score, value_new(-1))), value_new(1e-4));
            Value* temp = value_add(loss_value, loss);
            value_free(loss_value);
            loss_value = temp;
            
            for (int j = 0; j < INPUT_SIZE; j++) {
                value_free(inputs[j]);
            }
            free(inputs);
            value_free(label);
            value_free(score);
            value_free(output);
            value_free(loss);
        }
        Value* batch_loss = value_div(loss_value, value_new(BATCH_SIZE));
        value_free(loss_value);
        loss_value = batch_loss;

        printf("Loss calculated for epoch %d\n", epoch);
        print_value(loss_value);

        // Backward pass
        backward(loss_value);

        printf("Backward pass completed for epoch %d\n", epoch);

        // Update weights
        double learning_rate = 1.0 - 0.9 * epoch / 100.0;
        for (int i = 0; i < model->nlayers; i++) {
            for (int j = 0; j < model->layers[i]->nout; j++) {
                Neuron* neuron = model->layers[i]->neurons[j];
                for (int k = 0; k < neuron->nin; k++) {
                    neuron->w[k]->data -= learning_rate * neuron->w[k]->grad;
                    neuron->w[k]->grad = 0;  // Reset gradient
                }
                neuron->b->data -= learning_rate * neuron->b->grad;
                neuron->b->grad = 0;  // Reset gradient
            }
        }

        printf("Weights updated for epoch %d\n", epoch);

        // Calculate accuracy
        int correct = 0;
        for (int i = 0; i < BATCH_SIZE; i++) {
            Value** inputs = malloc(INPUT_SIZE * sizeof(Value*));
            if (inputs == NULL) {
                fprintf(stderr, "Memory allocation failed for inputs\n");
                mlp_free(model);
                value_free(loss_value);
                return 1;
            }
            for (int j = 0; j < INPUT_SIZE; j++) {
                inputs[j] = value_new(X[i][j]);
            }
            Value* output = mlp_call(model, inputs);
            if (output == NULL) {
                fprintf(stderr, "mlp_call returned NULL during accuracy calculation\n");
                mlp_free(model);
                value_free(loss_value);
                for (int j = 0; j < INPUT_SIZE; j++) {
                    value_free(inputs[j]);
                }
                free(inputs);
                return 1;
            }
            int pred = output->data > 0 ? 1 : 0;
            if (pred == y[i]) {
                correct++;
            }
            for (int j = 0; j < INPUT_SIZE; j++) {
                value_free(inputs[j]);
            }
            free(inputs);
            value_free(output);
        }
        double accuracy = (double)correct / BATCH_SIZE;

        printf("Epoch %d, loss: %f, accuracy: %f\n", epoch, loss_value->data, accuracy);

        value_free(loss_value);
    }

    // Free resources
    mlp_free(model);

    return 0;
}
