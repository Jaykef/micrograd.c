#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nn.c"

#define NUM_SAMPLES 100
#define NUM_FEATURES 2

typedef struct {
    double x[NUM_FEATURES];
    int y;
} Sample;

Sample* load_data(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    Sample* data = malloc(NUM_SAMPLES * sizeof(Sample));
    if (data == NULL) {
        printf("Memory allocation failed.\n");
        exit(1);
    }

    char line[1024];
    int i = 0;

    // Skip header
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file) && i < NUM_SAMPLES) {
        if (sscanf(line, "%lf,%lf,%d", &data[i].x[0], &data[i].x[1], &data[i].y) != 3) {
            printf("Error reading line %d\n", i+1);
            exit(1);
        }
        i++;
    }

    fclose(file);
    printf("Loaded %d samples\n", i);
    return data;
}

Value* loss(MLP* model, Sample* data, int batch_size) {
    Value* total_loss = value_new(0);
    int correct = 0;

    for (int i = 0; i < batch_size; i++) {
        Value* inputs[NUM_FEATURES];
        for (int j = 0; j < NUM_FEATURES; j++) {
            inputs[j] = value_new(data[i].x[j]);
        }

        Value* score = mlp_call(model, inputs);
        if (score == NULL) {
            printf("Error: mlp_call returned NULL for sample %d\n", i);
            exit(1);
        }

        Value* y = value_new(data[i].y * 2.0 - 1.0);
        Value* loss = value_relu(value_add(value_new(1), value_mul(value_new(-1), value_mul(y, score))));
        total_loss = value_add(total_loss, loss);

        if ((data[i].y == 1 && score->data > 0) || (data[i].y == 0 && score->data <= 0)) {
            correct++;
        }

        for (int j = 0; j < NUM_FEATURES; j++) {
            value_free(inputs[j]);
        }
        value_free(y);
        value_free(score);
        value_free(loss);
    }

    total_loss = value_mul(total_loss, value_new(1.0 / batch_size));

    printf("Loss: %f, Accuracy: %f%%\n", total_loss->data, (double)correct / batch_size * 100);

    return total_loss;
}

void zero_grad(MLP* model) {
    for (int i = 0; i < model->nlayers; i++) {
        for (int j = 0; j < model->layers[i]->nout; j++) {
            for (int k = 0; k < model->layers[i]->nin; k++) {
                model->layers[i]->neurons[j]->w[k]->grad = 0;
            }
            model->layers[i]->neurons[j]->b->grad = 0;
        }
    }
}

void update_parameters(MLP* model, double learning_rate) {
    for (int i = 0; i < model->nlayers; i++) {
        for (int j = 0; j < model->layers[i]->nout; j++) {
            for (int k = 0; k < model->layers[i]->nin; k++) {
                model->layers[i]->neurons[j]->w[k]->data -= learning_rate * model->layers[i]->neurons[j]->w[k]->grad;
            }
            model->layers[i]->neurons[j]->b->data -= learning_rate * model->layers[i]->neurons[j]->b->grad;
        }
    }
}

void free_mlp(MLP* model) {
    for (int i = 0; i < model->nlayers; i++) {
        for (int j = 0; j < model->layers[i]->nout; j++) {
            for (int k = 0; k < model->layers[i]->nin; k++) {
                value_free(model->layers[i]->neurons[j]->w[k]);
            }
            value_free(model->layers[i]->neurons[j]->b);
            free(model->layers[i]->neurons[j]->w);
            free(model->layers[i]->neurons[j]);
        }
        free(model->layers[i]->neurons);
        free(model->layers[i]);
    }
    free(model->layers);
    free(model);
}

int main(void) {
    srand(time(NULL));

    Sample* data = load_data("data/make_moons.csv");

    int nouts[] = {16, 16, 1};
    MLP* model = mlp_new(NUM_FEATURES, nouts, 3);
    if (model == NULL) {
        printf("Error: Failed to create MLP\n");
        exit(1);
    }

    int num_epochs = 100; 
    int batch_size = NUM_SAMPLES;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Starting epoch %d\n", epoch);
        
        Value* total_loss = loss(model, data, batch_size);
        if (total_loss == NULL) {
            printf("Error: loss function returned NULL\n");
            exit(1);
        }
        
        zero_grad(model);
        backward(total_loss);
        
        double learning_rate = 1.0 - 0.9 * epoch / num_epochs;
        update_parameters(model, learning_rate);

        value_free(total_loss);
        
        printf("Epoch %d completed\n", epoch);
    }

    // Free memory
    free(data);
    free_mlp(model);

    printf("Training completed successfully\n");
    return 0;
}
