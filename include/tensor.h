#ifndef TENSOR_H
#define TENSOR_H

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

namespace tensor
{
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

        Strides calculate_strides()
        {
            /*
            Construct stride tensor for indexing.
            */
            if (shape.empty())
                return {0};

            size_t num_dims = shape.size();
            Strides stride;
            stride.resize(num_dims);

            if (num_dims > 0)
            {
                stride[num_dims - 1] = 1;

                // Index backwards for stride calculation
                for (size_t i = num_dims - 1; i > 0; --i)
                {
                    stride[i - 1] = stride[i] * shape[i];
                }
            }
            return stride;
        };

    public:
        /*
        Constructors go here:
        */

        // Constructor for when everything is defined (rarely used.)
        Tensor(const Shape s, const Data<T> d, const Strides st) : shape(s), data(d), stride(st) {

                                                                   };

        // Constructor for when shape and data is defined:
        Tensor(const Shape s, const Data<T> d) : shape(s), data(d)
        {
            size_t total_size = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());

            // Verify data size and intialized shape are equal. 
            if (!(data.size() == total_size)){
                cout << "Total size of data does not match shape. \n"
                << "Data size: " << data.size() << "\n"
                << "Total size (inferred from shape): " << total_size << endl; 
                // End here since it's not verified.
                assert(data.size() == total_size);
            };

            stride = calculate_strides();
        };

        // Constructor is an empty tensor of specified shape:
        Tensor(const Shape shape) : shape(shape)
        {
            stride = calculate_strides();
            size_t total_size = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
            data.resize(total_size, 0);
        };

        /*
        Utility functions go here:
        */
        Shape get_shape() const
        {
            return shape;
        };

        Tensor<T> t()
        {
            Strides reverse_stride(stride.rbegin(), stride.rend());
            Shape reverse_shape(shape.rbegin(), shape.rend());
            Tensor<T> transposed_tensor(reverse_shape, data, reverse_stride);
            return transposed_tensor;
        };

        /*
        Overloading operators go here:
        */

        // Overload indexing:
        // TODO: Add int-based indexing
        T operator[](Index &index) const
        {
            size_t n = index.size();
            int global_idx = 0;
            for (size_t i = 0; i < n; i++)
            {
                global_idx += index[i] * stride[i];
            };
            return data[global_idx];
        };

        // Overload addition (Scalar addition)
        Tensor<T> operator+(const T &scalar)
        {
            Shape self_shape = shape;
            Data<T> output_data(data.size());
            for (size_t i = 0; i < data.size(); i++)
            {
                output_data[i] = data[i] + scalar;
            }
            Tensor<T> output(self_shape, output_data);
            return output;
        }

        // Overload addition
        Tensor<T> operator+(const Tensor<T> &t)
        {
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
            for (size_t i = 0; i < data.size(); i++)
            {
                Index ii = {i};
                output_data[i] = data[i] + t[ii];
            };

            Tensor<T> output(self_shape, output_data);
            return output;
        };

        // Overload multiplication (Scalar multiplication)
        Tensor<T> operator*(const T &scalar)
        {
            Shape self_shape = shape;
            Data<T> output_data(data.size());
            for (size_t i = 0; i < data.size(); i++)
            {
                output_data[i] = data[i] * scalar;
            }
            Tensor<T> output(self_shape, output_data);
            return output;
        }

        // Overload multiplication (Hadamard product)
        Tensor<T> operator*(const Tensor<T> &t)
        {
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
            for (size_t i = 0; i < data.size(); i++)
            {
                Index ii = {i};
                output_data[i] = data[i] * t[ii];
            };

            Tensor<T> output(self_shape, output_data);
            return output;
        };
    };
} // End of tensor namespace

#endif // TENSOR_H