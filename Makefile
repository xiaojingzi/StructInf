CC = g++
COPTS = -Wno-deprecated -O3 -march=native -std=c++11

structinf: StructInfCounter.cpp
	$(CC) $(COPTS) StructInfCounter.cpp -o structinf

clean:
	rm -f *.o structinf