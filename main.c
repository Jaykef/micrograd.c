#include <stdio.h>
#include "test/test.h"
#include "micrograd.c/nn.h"
#include "micrograd.c/engine.h"

int main(void) {
    test_sanity_check();
    test_more_ops();

    printf("All tests passed!\n");
    return 0;
}
