#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "tensor.h"
#include "doctest.h"
#include <iostream>

using namespace std;
using namespace tensor;

TEST_CASE("Tensor Construction (Shape Only)"){
    Tensor<int> a({2});
};

TEST_CASE("Tensor Construction (Shape and Data)"){
    Tensor<int> a({2}, {1, 2});
};

TEST_CASE("Tensor Indexing"){
    // TODO: Clean this up such that
    // we can also use int or size_t
    Tensor<int> a({2}, {1, 2});
    Index zeroth = {0};
    Index first = {1};

    CHECK(a[zeroth] == 1);
    CHECK(a[first] == 2);
};

TEST_CASE("Scalar Addition"){
    // When both are int:
    Tensor<int> a({2}, {1, 2});
    int b = 3;
    Tensor<int> c = a + b;

    Index zeroth = {0};
    Index first = {1};

    CHECK(c[zeroth] == 4);
    CHECK(c[first] == 5);

    // When scalar is a different dtype:
    double b_double = 3;
    Tensor<int> c_double = a + b_double;

    CHECK(c_double[zeroth] == 4);
    CHECK(c_double[first] == 5);
};

TEST_CASE("Tensor Addition"){
    Tensor<int> a({2}, {1, 2});
    Tensor<int> b({2}, {2, 2});
    Tensor<int> c = a + b;

    Index zeroth = {0};
    Index first = {1};

    CHECK(c[zeroth] == 3);
    CHECK(c[first] == 4);
};

TEST_CASE("Scalar Multiplication"){
    // When both are int:
    Tensor<int> a({2}, {1, 2});
    int b = 3;
    Tensor<int> c = a * b;

    Index zeroth = {0};
    Index first = {1};

    CHECK(c[zeroth] == 3);
    CHECK(c[first] == 6);

    // When scalar is a different dtype:
    double b_double = 3;
    Tensor<int> c_double = a * b_double;

    CHECK(c_double[zeroth] == 3);
    CHECK(c_double[first] == 6);
};

TEST_CASE("Hadamard Product"){
    Tensor<int> a({2}, {1, 2});
    Tensor<int> b({2}, {2, 6});
    Tensor<int> c = a * b;

    Index zeroth = {0};
    Index first = {1};

    CHECK(c[zeroth] == 2);
    CHECK(c[first] == 12);
};

TEST_CASE("Transpose"){
    Tensor<int> a({2,2}, {1,2,3,4});
    Tensor<int> b = a.t();

    Index i = {0, 1};
    Index j = {1, 0};

    CHECK(a[i] == b[j]);
};