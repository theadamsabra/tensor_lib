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
    Tensor<int> a({2}, {1, 2});
    Index zeroth = {0};
    Index first = {1};

    CHECK(a[zeroth] == 1);
    CHECK(a[first] == 2);
};