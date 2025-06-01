CC = gcc
NVCC = nvcc
CFLAGS = -march=native -O1 -ffast-math -ftree-vectorize -funroll-loops -fopenmp
NVCCFLAGS = -arch=sm_61 -O3 --use_fast_math
CPPFLAGS = -DN=2048 -DSEED=0
LDFLAGS = -lm -lcudart

TARGET = tiny_md
OBJS = tiny_md.o core.o core_cuda.o wtime.o mtwister.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile CUDA code
core_cuda.o: core.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

# Compile CPU code
core.o: core.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o
