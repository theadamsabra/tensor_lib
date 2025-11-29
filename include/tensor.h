#include <vector>
#include <iostream>
#include <optional>
#include <tuple>
#include <numeric>
#include <cassert>
#include <functional>
#include <algorithm>
#include <variant>
using namespace std;

template <typename T>

using Data = vector<T>;
using Shape = vector<size_t>;
using Strides = vector<size_t>;
using Index = vector<size_t>;
using N = size_t;

template <typename T>
class Tensor 
{
    /*
    Core Tensor type.

    @param Shape shape: shape of tensor
    @param Data<T> data: data within tensor. Datatype must be specified.
    @param Strides stride: data within tensor
    */

private:
    Shape shape;
    Data<T> data;
    Strides stride;

    void calculate_strides(){
        /*
        Construct stride tensor for indexing. 
        */
        if (shape.empty()) return;

        size_t num_dims = shape.size();
        Strides stride(num_dims);

        for (int i=0; i<num_dims; i++){
            if (i != num_dims - 1){
                stride[i] = accumulate(&shape[i + 1], &shape[num_dims], 1, multiplies<size_t>());
            } else {
                stride[i] = 1;
            };

        reverse(stride.begin(), stride.end());
        };
    };

public:
    // Default constructor is an emtpy tensor of specified shape:
    Tensor(const Shape shape) : shape(shape)
    {
        calculate_strides();
        size_t total_size = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
        data.resize(total_size);
    };

    Shape get_shape(){
        return shape;
    };
    // Overload indexing:
    T& operator[](Index& index){
        size_t n = index.size();
        int global_idx = 0; 
        for (int i=0; i<n; i++){
            global_idx += index[i] * stride[i];
        };
        return data[global_idx];
    };

    // Read-only indexing:
    const T& operator[](Index& index) const {
        size_t n = index.size();
        int global_idx = 0; 
        for (int i=0; i<n; i++){
            global_idx += index[i] * stride[i];
        };
        return data[global_idx];
    };

    // Overload addition
    Tensor<T&> operator+(Tensor<T&> t){
        // Verify shape is equal:
        Shape self_shape = shape; 
        Shape additive_shape = t.get_shape();
        assert(self_shape == additive_shape);

        // Verify rank is equal:
        size_t n = self_shape.size();
        size_t m = additive_shape.size();
        assert(n == m);

        Tensor<T> output(self_shape);
        // Add by simple indexing across entire vector:
        for (int i=0; i<data.size(); i++){
            output[i] = data[i] + additive_shape[i];
        };
        return output;
    };

    // Overload multiplication 
    Tensor<T&> operator*(Tensor<T&> t){
        // Verify shape is equal:
        Shape self_shape = shape; 
        Shape additive_shape = t.get_shape();
        assert(self_shape == additive_shape);

        // Verify rank is equal:
        size_t n = self_shape.size();
        size_t m = additive_shape.size();
        assert(n == m);

        Tensor<T> output(self_shape);
        // Add by simple indexing across entire vector:
        for (int i=0; i<data.size(); i++){
            output[i] = data[i] * additive_shape[i];
        };
        return output;
    };
};

int main(){
    Tensor<float> t({2,3,4});
    return 0;
};