# Basic Tensor Ops

## Tensor Formalization and Indexing

### Tensor Definition and Rank

A **Tensor** $\mathbf{A}$ is a multidimensional array holding elements of a certain type, $T$.

* **Rank (or Order), $N$:** The number of dimensions.
* **Shape, $\mathbf{D}$:** A vector of dimension sizes.
    $$\mathbf{D} = (d_0, d_1, \dots, d_{N-1})$$
    where $d_i$ is the extent (size) of the $i$-th dimension.
* **Total Number of Elements, $L$:** The product of all dimensions.
    $$L = \prod_{i=0}^{N-1} d_i = d_0 \times d_1 \times \dots \times d_{N-1}$$
* **Storage:** Data is stored in a single, **contiguous 1D array** of length $L$.

---

## Strides Calculation (Row-Major Order)

The **Stride Vector**, $\mathbf{S}$, defines the memory jump needed to move one step along each dimension. This uses **Row-Major Order** (C-style), where the last dimension ($d_{N-1}$) is fastest changing.

### Formal Definition:

The stride $s_i$ for dimension $d_i$ is calculated as:

$$
s_i = \begin{cases}
1 & \text{if } i = N-1 \quad (\text{Last dimension}) \\
\prod_{j=i+1}^{N-1} d_j & \text{if } 0 \le i < N-1
\end{cases}
$$

### Example (Shape $\mathbf{D} = (2, 3, 4)$):

The stride vector is $\mathbf{S} = (12, 4, 1)$.

| Dimension $i$ | $d_i$ | Calculation | Stride $s_i$ |
| :---: | :---: | :---: | :---: |
| 2 | 4 | $s_2 = 1$ | 1 |
| 1 | 3 | $s_1 = d_2 = 4$ | 4 |
| 0 | 2 | $s_0 = d_1 \times d_2 = 3 \times 4$ | 12 |

---

## Indexing Formula

Given the coordinate vector $\mathbf{I} = (i_0, i_1, \dots, i_{N-1})$, the **Flat Array Index** $k$ is calculated as the **dot product** of the coordinate vector $\mathbf{I}$ and the stride vector $\mathbf{S}$.

### Formal Definition:

$$
k = \text{Flat Index} = \sum_{j=0}^{N-1} (i_j \cdot s_j)
$$

### Example (Coordinates $\mathbf{I} = (1, 0, 2)$):

Using Strides $\mathbf{S}=(12, 4, 1)$:

$$
k = (1 \cdot 12) + (0 \cdot 4) + (2 \cdot 1) = 14
$$
That's the next logical step! Element-wise operations are fundamental to tensor math.

Here is the formalization for **Tensor Addition** and **Tensor Multiplication** (Hadamard product), focusing on the necessary **Shape Constraint** first.

---

## The Shape Constraint (Broadcasting Excluded)

For basic element-wise operations (addition and multiplication), the two input tensors, $\mathbf{A}$ and $\mathbf{B}$, must have the exact same shape. If the shapes don't match, the operation is undefined in this basic context.

Let:
* $\mathbf{A}$ have Shape $\mathbf{D}_{\mathbf{A}} = (a_0, a_1, \dots, a_{N-1})$
* $\mathbf{B}$ have Shape $\mathbf{D}_{\mathbf{B}} = (b_0, b_1, \dots, b_{M-1})$

### Formal Constraint:

For $\mathbf{A} + \mathbf{B}$ or $\mathbf{A} \odot \mathbf{B}$ to be defined (without broadcasting):

1.  **Rank must be equal:** $N = M$.
2.  **All dimensions must be equal:** $a_i = b_i$ for all $i \in \{0, 1, \dots, N-1\}$.

The resulting tensor $\mathbf{C}$ will have the same shape: $\mathbf{D}_{\mathbf{C}} = \mathbf{D}_{\mathbf{A}}$.

---

## Element-wise Tensor Addition

**Tensor Addition** is defined by adding the corresponding elements of the two input tensors, $\mathbf{A}$ and $\mathbf{B}$, to produce the result tensor $\mathbf{C}$.

### Formal Definition:

Let $\mathbf{C} = \mathbf{A} + \mathbf{B}$. The element $c_{\mathbf{I}}$ at the coordinate $\mathbf{I}=(i_0, i_1, \dots, i_{N-1})$ in $\mathbf{C}$ is the sum of the elements $a_{\mathbf{I}}$ and $b_{\mathbf{I}}$ at the same coordinate in $\mathbf{A}$ and $\mathbf{B}$, respectively.

$$
c_{(i_0, i_1, \dots, i_{N-1})} = a_{(i_0, i_1, \dots, i_{N-1})} + b_{(i_0, i_1, \dots, i_{N-1})}
$$

### Implementation Focus: Flat Index Iteration

In code, this is best implemented by iterating through the **single flat 1D index** ($k$) from $0$ to $L-1$ (where $L$ is the total size) and performing the operation directly on the storage arrays.

$$
\mathbf{C}.data[k] = \mathbf{A}.data[k] + \mathbf{B}.data[k], \quad \text{for } k \in \{0, 1, \dots, L-1\}
$$

---

## Element-wise Tensor Multiplication (Hadamard Product)

**Element-wise Tensor Multiplication** (also known as the **Hadamard Product** or Schur Product) is defined by multiplying the corresponding elements of the two input tensors, $\mathbf{A}$ and $\mathbf{B}$, to produce the result tensor $\mathbf{C}$.

*Note: This is distinct from matrix multiplication $(\mathbf{A} \times \mathbf{B})$ or tensor contraction, which involve summing over shared dimensions.*

### Formal Definition:

Let $\mathbf{C} = \mathbf{A} \odot \mathbf{B}$ (where $\odot$ denotes the Hadamard product). The element $c_{\mathbf{I}}$ at coordinate $\mathbf{I}$ in $\mathbf{C}$ is the product of the corresponding elements in $\mathbf{A}$ and $\mathbf{B}$.

$$
c_{(i_0, i_1, \dots, i_{N-1})} = a_{(i_0, i_1, \dots, i_{N-1})} \cdot b_{(i_0, i_1, \dots, i_{N-1})}
$$

### Implementation Focus: Flat Index Iteration

Similar to addition, this is implemented efficiently by iterating over the flat 1D index ($k$).

$$
\mathbf{C}.data[k] = \mathbf{A}.data[k] \cdot \mathbf{B}.data[k], \quad \text{for } k \in \{0, 1, \dots, L-1\}
$$

---

## Scalar Operations

Scalar operations involve combining a **Tensor** $\mathbf{A}$ with a single **Scalar** value $\alpha$. These operations (addition, subtraction, multiplication, division) are performed **element-wise**, where the scalar is applied uniformly to every element in the tensor.

### Formal Constraint

Scalar operations are always defined, regardless of the tensor's shape or rank. The resulting tensor $\mathbf{C}$ has the exact same shape as the input tensor $\mathbf{A}$.

### Element-wise Scalar Addition

Let $\mathbf{C} = \mathbf{A} + \alpha$.

The element $c_{\mathbf{I}}$ at the coordinate $\mathbf{I}=(i_0, i_1, \dots, i_{N-1})$ in $\mathbf{C}$ is the sum of the element $a_{\mathbf{I}}$ and the scalar $\alpha$.

$$
c_{(i_0, i_1, \dots, i_{N-1})} = a_{(i_0, i_1, \dots, i_{N-1})} + \alpha
$$

### Element-wise Scalar Multiplication

Let $\mathbf{C} = \mathbf{A} \cdot \alpha$.

The element $c_{\mathbf{I}}$ at the coordinate $\mathbf{I}$ in $\mathbf{C}$ is the product of the element $a_{\mathbf{I}}$ and the scalar $\alpha$.

$$
c_{(i_0, i_1, \dots, i_{N-1})} = a_{(i_0, i_1, \dots, i_{N-1})} \cdot \alpha
$$

### Implementation Focus: Flat Index Iteration

For any scalar operation $(\star \in \{+,-,\cdot,/\})$:

$$
\mathbf{C}.data[k] = \mathbf{A}.data[k] \star \alpha, \quad \text{for } k \in \{0, 1, \dots, L-1\}
$$

---

## Tensor Reduction (Summation)

**Tensor Reduction** is the process of reducing the rank (number of dimensions) of a tensor by computing a summary statistic (like **sum**, mean, min, max) across one or more dimensions.

### Global Summation (Total Reduction)

The simplest form is computing the sum of **all elements** in the tensor $\mathbf{A}$, resulting in a single scalar value $\sigma$.

$$
\sigma = \sum_{\mathbf{I}} a_{\mathbf{I}} = \sum_{k=0}^{L-1} \mathbf{A}.data[k]
$$
where the first sum is over all valid coordinate vectors $\mathbf{I}$.

### Summation Along a Specific Axis (Dimension)

Reducing the tensor along a specified axis $j$ (where $0 \le j < N$) results in a new tensor $\mathbf{C}$ with **Rank $N-1$**.

Let $\mathbf{A}$ have shape $\mathbf{D}=(d_0, \dots, d_{N-1})$. The result tensor $\mathbf{C}$ has the shape of $\mathbf{D}$ with the $j$-th dimension removed.

#### Formal Definition

The element $c_{\mathbf{I}'}$ in the result tensor $\mathbf{C}$ is the sum of all elements in $\mathbf{A}$ that share the same coordinates for all dimensions *except* dimension $j$.

Let $\mathbf{I} = (i_0, \dots, i_j, \dots, i_{N-1})$ be a coordinate in $\mathbf{A}$. Let $\mathbf{I}' = (i_0, \dots, i_{j-1}, i_{j+1}, \dots, i_{N-1})$ be the corresponding coordinate in $\mathbf{C}$.

$$
c_{\mathbf{I}'} = \sum_{i_j=0}^{d_j-1} a_{(i_0, \dots, i_j, \dots, i_{N-1})}
$$

#### Example (Rank 3 Tensor, Shape $(2, 3, 4)$, Sum over Axis 1)

If $\mathbf{A}$ has shape $(d_0, d_1, d_2) = (2, 3, 4)$, summing over axis $j=1$ (size $d_1=3$) results in a tensor $\mathbf{C}$ with shape $(2, 4)$.

The element $c_{(i_0, i_2)}$ is calculated as:

$$
c_{(i_0, i_2)} = a_{(i_0, 0, i_2)} + a_{(i_0, 1, i_2)} + a_{(i_0, 2, i_2)}
$$
