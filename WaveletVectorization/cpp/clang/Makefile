all: prog clean
.PHONY: clean

CC=clang++
CFLAGS=-Wall -O2 -std=c++11

prog: vectorizer.o solver.o
	$(CC) $(CFLAGS) -o prog vectorizer.o solver.o

vectorizer.o: CubicBezier.hpp vectorizer.cpp
	$(CC) $(CFLAGS) -c vectorizer.cpp

solver.o: solver.cpp
	$(CC) $(CFLAGS) -c solver.cpp

clean:
	rm -f vectorizer.o solver.o
