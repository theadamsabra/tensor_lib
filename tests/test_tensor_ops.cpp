#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "tensor.h"
#include "doctest.h"
#include <iostream>
using namespace std;

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

TEST_CASE("Tensor Addition"){
    Tensor<int> a({2}, {1, 2});
    Tensor<int> b({2}, {2, 2});
    Tensor<int> c = a + b;

    Index zeroth = {0};
    Index first = {1};

    CHECK(c[zeroth] == 3);
    CHECK(c[first] == 4);
};

TEST_CASE("Tensor Addition"){
    Tensor<int> a({2}, {1, 2});
    Tensor<int> b({2}, {2, 6});
    Tensor<int> c = a * b;

    Index zeroth = {0};
    Index first = {1};

    CHECK(c[zeroth] == 2);
    CHECK(c[first] == 12);
};