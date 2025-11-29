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

    Strides calculate_strides(){
        /*
        Construct stride tensor for indexing. 
        */
        if (shape.empty()) return {0};

        size_t num_dims = shape.size();
        Strides stride;
        stride.resize(num_dims);
        
        if (num_dims > 0){
            stride[num_dims - 1] = 1;
            
            // Index backwards for stride calculation
            for (size_t i = num_dims - 1; i>0; --i){
                stride[i - 1] = stride[i] * shape[i];
            }
        }
        return stride;
    };

public:
    // Constructor for when shape and data is defined:
    Tensor(const Shape s, const Data<T> d) : shape(s), data(d)
    {
        size_t total_size = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
        // TODO: Check shape and data are same size.
        stride = calculate_strides();
    };

    // Constructor is an emtpy tensor of specified shape:
    Tensor(const Shape shape) : shape(shape)
    {
        stride = calculate_strides();
        size_t total_size = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
        data.resize(total_size, 0);
    };

    Shape get_shape(){
        return shape;
    };

    // Overload indexing:
    T operator[](Index &index){
        size_t n = index.size();
        int global_idx = 0; 
        for (int i=0; i<n; i++){
            // TODO: Check why stride isn't being constructed
            global_idx += index[i] * stride[i];
        };
        return data[global_idx];
    };

    // Overload addition
    Tensor<T> operator+(Tensor<T> &t){
        // Verify shape is equal:
        Shape self_shape = shape; 
        Shape additive_shape = t.get_shape();
        assert(self_shape == additive_shape);

        // Verify rank is equal:
        size_t n = self_shape.size();
        size_t m = additive_shape.size();
        assert(n == m);

        Data<T> output_data(data.size());

        // Add by simple indexing across entire vector:
        for (size_t i=0; i<data.size(); i++){
            Index ii = {i};
            output_data[i] = data[i] + t[ii];
        };

        Tensor<T> output(self_shape, output_data);
        return output;
    };

    // Overload multiplication (Scalar multiplication)
    Tensor<T> operator*(T scalar){
        Shape self_shape = shape; 
        Data<T> output_data(data.size());
        for (size_t i=0; i<data.size(); i++){
            output_data[i] = data[i] * scalar;
        }
        Tensor<T> output(self_shape, output_data);
        return output;
    }

    // Overload multiplication (Hadamard product)
    Tensor<T> operator*(Tensor<T> t){
        // Verify shape is equal:
        Shape self_shape = shape; 
        Shape additive_shape = t.get_shape();
        assert(self_shape == additive_shape);

        // Verify rank is equal:
        size_t n = self_shape.size();
        size_t m = additive_shape.size();
        assert(n == m);

        Data<T> output_data(data.size());

        // Add by simple indexing across entire vector:
        for (size_t i=0; i<data.size(); i++){
            Index ii = {i};
            output_data[i] = data[i] * t[ii];
        };

        Tensor<T> output(self_shape, output_data);
        return output;
    };
};