CC = nvcc 

FLAGS = --std=c++11 -Werror cross-execution-space-call -lm
FILES = main.cu

PROG = main

all:
	$(CC) $(FLAGS) -o $(PROG) $(FILES)

clean:
	rm -f *.o $(PROG)