#include "micrograd.c/nn.h"
#include "micrograd.c/engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_SAMPLES 100
#define INPUT_SIZE 2
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 1

void read_data(const char* filename, double X[N_SAMPLES][INPUT_SIZE], int y[N_SAMPLES]) {
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

int main(void) {
    srand(time(NULL));

    // Read data from CSV file
    double X[N_SAMPLES][INPUT_SIZE];
    int y[N_SAMPLES];
    read_data("data/make_moons.csv", X, y);

    // Initialize model
    int layer_sizes[] = {INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    MLP* model = mlp_new(INPUT_SIZE, &layer_sizes[1], 3);

    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        double total_loss = 0.0;
        int correct = 0;

        for (int i = 0; i < N_SAMPLES; i++) {
            // Forward pass
            Value* inputs[INPUT_SIZE];
            for (int j = 0; j < INPUT_SIZE; j++) {
                inputs[j] = value_new(X[i][j]);
            }
            Value* output = mlp_call(model, inputs);
            
            // Compute loss
            Value* target = value_new(y[i] * 2.0 - 1.0);  // Convert 0/1 to -1/+1
            Value* loss = value_relu(value_add(value_mul(value_neg(target), output), value_new(1.0)));
            
            total_loss += loss->data;
            
            // Compute accuracy
            if ((y[i] == 1 && output->data > 0) || (y[i] == 0 && output->data <= 0)) {
                correct++;
            }

            // Backward pass
            backward(loss);

            // Update weights
            double learning_rate = 0.01;
            mlp_update(model, learning_rate);

            // Free memory
            for (int j = 0; j < INPUT_SIZE; j++) {
                value_free(inputs[j]);
            }
            value_free(output);
            value_free(target);
            value_free(loss);
        }
        
        double avg_loss = total_loss / N_SAMPLES;
        double accuracy = (double)correct / N_SAMPLES * 100.0;
        printf("step %d loss %f, accuracy %f%%\n", epoch, avg_loss, accuracy);

        mlp_zero_grad(model);
    }

    // Free model
    mlp_free(model);

    return 0;
}
