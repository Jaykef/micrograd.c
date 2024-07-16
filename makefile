CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99 -I./micrograd.c -I./test
LDFLAGS = -lm

SOURCES = main.c train.c micrograd.c/engine.c micrograd.c/nn.c test/test.c
OBJECTS = $(SOURCES:.c=.o)
EXECUTABLES = main train

.PHONY: all clean

all: $(EXECUTABLES)

main: main.o micrograd.c/engine.o micrograd.c/nn.o test/test.o
	$(CC) $^ $(LDFLAGS) -o $@

train: train.o micrograd.c/engine.o micrograd.c/nn.o
	$(CC) $^ $(LDFLAGS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLES)
