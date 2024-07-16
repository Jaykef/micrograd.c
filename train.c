#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "micrograd.c/nn.h"
#include "micrograd.c/engine.h"

#define N_SAMPLES 200
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 1

void load_data(const char* filename, double X[N_SAMPLES][2], int y[N_SAMPLES]) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file);  // Skip header

    for (int i = 0; i < N_SAMPLES; i++) {
        if (fscanf(file, "%lf,%lf,%d", &X[i][0], &X[i][1], &y[i]) != 3) {
            printf("Error reading line %d\n", i + 1);
            exit(1);
        }
    }

    fclose(file);
}

double loss(MLP* model, double X[N_SAMPLES][2], int y[N_SAMPLES], double* accuracy) {
    double total_loss = 0.0;
    int correct = 0;
    
    mlp_zero_grad(model);

    for (int i = 0; i < N_SAMPLES; i++) {
        // Forward pass
        Value* inputs[2];
        for (int j = 0; j < 2; j++) {
            inputs[j] = value_new(X[i][j]);
        }
        Value* output = mlp_call(model, inputs);
        
        // Compute margin loss
        Value* target = value_new(y[i] * 2.0 - 1.0);  // Convert 0/1 to -1/+1
        Value* margin_loss = value_relu(value_add(value_neg(value_mul(target, output)), value_new(1.0)));
        total_loss += margin_loss->data;

        // Accumulate gradients
        backward(margin_loss);

        // Compute accuracy
        if ((y[i] == 1 && output->data > 0) || (y[i] == 0 && output->data <= 0)) {
            correct++;
        }

        // Free memory
        for (int j = 0; j < 2; j++) {
            value_free(inputs[j]);
        }
        value_free(output);
        value_free(target);
        value_free(margin_loss);
    }

    // Compute regularization loss and add to total loss
    double reg_loss = 0.0;
    double alpha = 1e-4;
    Value** params = mlp_parameters(model);
    int param_count = mlp_parameters_count(model);
    for (int p = 0; p < param_count; p++) {
        reg_loss += params[p]->data * params[p]->data;
    }
    reg_loss *= alpha;
    total_loss += reg_loss;

    // Average loss over samples
    total_loss /= N_SAMPLES;

    // Compute accuracy
    *accuracy = (double)correct / N_SAMPLES * 100.0;

    // Free parameter array
    free(params);

    return total_loss;
}


int main(void) {
    srand(time(NULL));

    // laod data
    double X[N_SAMPLES][2];
    int y[N_SAMPLES];
    load_data("data/make_moons.csv", X, y);

    // Initialize model
    int layer_sizes[] = {2, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    MLP* model = mlp_new(2, &layer_sizes[1], 3);

    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        double accuracy = 0.0;
        double avg_loss = loss(model, X, y, &accuracy);

        // Update weights
        double learning_rate = 0.01;
        mlp_update(model, learning_rate);

        // Print progress
        printf("step %d loss %f, accuracy %f%%\n", epoch, avg_loss, accuracy);

        // Zero out gradients for next epoch
        mlp_zero_grad(model);
    }

    // Free model
    mlp_free(model);

    return 0;
}
