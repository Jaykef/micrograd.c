#include <stdio.h>
#include "test/test.c"

int main(void) {
    test_sanity_check();
    test_more_ops();

    printf("All tests passed!\n");
    return 0;
}
