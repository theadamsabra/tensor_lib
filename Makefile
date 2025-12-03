CC = g++
CFLAGS = -I include -g -Wall -Wextra -pedantic -Werror 

all: main test_tensor_ops

main: src/main.cpp
	$(CC) $(CFLAGS) src/main.cpp -o main && ./main 

test_tensor_ops: tests/test_tensor_ops.cpp 
	$(CC) $(CFLAGS) tests/test_tensor_ops.cpp -o test_tensor_ops && ./test_tensor_ops

clean:
	rm -f main
	rm -f test_tensor_ops