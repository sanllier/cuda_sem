all: mul
OBJECTS = parparser.o main.o kernel.o
DFLAG =

mul: $(OBJECTS)
	g++ -L/opt/cuda/lib64 $^ -lcudart -o $@

main.o: src/main.cpp
	g++ -D$(DFLAG) -Iinclude -I../lib/parparser -I/opt/cuda/include $^ -c -o $@

parparser.o: ../lib/parparser/parparser.cpp
	g++ -std=c++0x -c $^

kernel.o: src/kernel.cu
	nvcc -arch=sm_20 -Iinclude $^ -c -o $@	

clean:
	rm -rf *.o mul 
