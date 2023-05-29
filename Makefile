CC := nvcc
CFLAGS := -std=c++17 -Iinclude

SOURCES := heat.cu include/heatmap.c include/lodepng.cpp
OBJECTS := $(SOURCES:.cu=.o)
EXECUTABLE := heat

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(EXECUTABLE)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
