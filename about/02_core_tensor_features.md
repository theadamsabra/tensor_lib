# Core Tensor Features

## Transpose ($\mathbf{A}^{\text{T}}$)

Transpose is essential for manipulating matrices and higher-rank tensors, especially when preparing for true matrix multiplication (which relies on aligning dimensions).

### Implementation Principle

The most efficient way to implement transpose for a flat-array tensor is to **not move the data**. Instead, you create a new Tensor $\mathbf{C}$ that reinterprets the original data by swapping its metadata.

| Metadata Component | Action for Transpose |
| :--- | :--- |
| **Data Vector** | $\mathbf{C}$ shares the **exact same memory** as $\mathbf{A}$'s data. |
| **Shape $\mathbf{D}$** | The dimensions are **reversed** (e.g., $(2, 3, 4) \to (4, 3, 2)$). |
| **Strides $\mathbf{S}$** | The strides are also **reversed** (e.g., $(12, 4, 1) \to (1, 4, 12)$). |

### Mathematical Formalization (2D Matrix Example)

If $\mathbf{A}$ is an $M \times N$ matrix with coordinates $(i, j)$, the transpose $\mathbf{C} = \mathbf{A}^{\text{T}}$ is an $N \times M$ matrix with coordinates $(j, i)$.

The core relationship is that the element at the new coordinate $(j, i)$ in $\mathbf{C}$ is the element at the original coordinate $(i, j)$ in $\mathbf{A}$.

$$
c_{(j, i)} = a_{(i, j)}
$$

By swapping the shape and strides, the flat index calculation automatically handles this mapping:

$$\text{Flat Index } k = (i \cdot s_i) + (j \cdot s_j)$$

When the shape and strides are reversed for $\mathbf{C}$, the new index $(j', i')$ correctly maps to the old index $(i, j)$ in the shared data array.

That's an excellent choice. Implementing the transpose efficiently by manipulating metadata is a key differentiator for a high-performance tensor library.

Here is the mathematical and logical formalization for implementing the **Transpose operation ($\mathbf{A}^{\text{T}}$)**, specifically using the **metadata manipulation** approach.

---

### Goal and Resulting Tensor

The goal of the transpose operation is to create a new Tensor, $\mathbf{C} = \mathbf{A}^{\text{T}}$, that reinterprets the order of the dimensions of the original tensor $\mathbf{A}$.

#### A. Core Principle (Data Sharing)

The underlying storage ($\mathbf{V}_{\mathbf{A}}$) of the original tensor $\mathbf{A}$ **is not copied or modified**. The new tensor $\mathbf{C}$ will point to the exact same data vector.
$$
\mathbf{C}.data = \mathbf{A}.data
$$

#### B. Resulting Shape

The shape $\mathbf{D}_{\mathbf{C}}$ of the resulting tensor $\mathbf{C}$ is the **reversed order** of the shape $\mathbf{D}_{\mathbf{A}}$ of the original tensor $\mathbf{A}$.

Let $\mathbf{D}_{\mathbf{A}} = (d_0, d_1, \dots, d_{N-1})$.

$$
\mathbf{D}_{\mathbf{C}} = (d_{N-1}, d_{N-2}, \dots, d_0)
$$

### Stride Calculation (The Key Step)

Since the underlying data is not moved, the new strides ($\mathbf{S}_{\mathbf{C}}$) must be calculated from the original strides ($\mathbf{S}_{\mathbf{A}}$) to correctly map the new reversed coordinates to the old data.

#### A. Original Stride Vector

Let the original stride vector for $\mathbf{A}$ be $\mathbf{S}_{\mathbf{A}} = (s_0, s_1, \dots, s_{N-1})$.

#### B. Resulting Stride Vector

The stride vector for $\mathbf{C}$ is simply the **reversed order** of the original stride vector $\mathbf{S}_{\mathbf{A}}$.

Let $s'_i$ be the $i$-th stride of the new tensor $\mathbf{C}$.

$$
\mathbf{S}_{\mathbf{C}} = (s_{N-1}, s_{N-2}, \dots, s_0)
$$

$$
s'_i = s_{(N-1)-i} \quad \text{for } i \in \{0, 1, \dots, N-1\}
$$

### Indexing Validation

This is the most critical validation of the transpose method. We must confirm that the new indexing formula correctly maps the new coordinates to the same flat index $k$.

#### A. Setup for a 2D Matrix (Rank 2)

* **Original (A):** Shape $(d_0, d_1)$, Strides $(s_0, s_1)$.
* **New (C):** Shape $(d_1, d_0)$, Strides $(s_1, s_0)$.
* **Original Coordinates:** $\mathbf{I}_{\mathbf{A}} = (i, j)$
* **New Coordinates:** $\mathbf{I}_{\mathbf{C}} = (j, i)$

#### B. Index Calculation

We must show that the flat index $k$ calculated using $\mathbf{I}_{\mathbf{A}}$ and $\mathbf{S}_{\mathbf{A}}$ equals the flat index $k'$ calculated using $\mathbf{I}_{\mathbf{C}}$ and $\mathbf{S}_{\mathbf{C}}$.

1.  **Original Flat Index ($k$):**
    $$
    k = (i \cdot s_0) + (j \cdot s_1)
    $$

2.  **New Flat Index ($k'$):**
    $$
    k' = (j \cdot s_1) + (i \cdot s_0)
    $$

Since addition is commutative ($a+b = b+a$), it is immediately clear that $k = k'$. This confirms that by simply reversing the shape and the strides, the new indexing rule automatically accesses the original element at the transposed position.

This concept holds true for arbitrary Rank $N$: the dot product remains the same regardless of the order of the paired terms.

### Implementation Logic Summary

The core implementation of the transpose method should follow these three steps:

1.  Create a new `Tensor` object $\mathbf{C}$.
2.  Set $\mathbf{C}$.shape by reversing $\mathbf{A}$.shape.
3.  Set $\mathbf{C}$.strides by reversing $\mathbf{A}$.strides.
4.  Point $\mathbf{C}$.data to $\mathbf{A}$.data.

This approach ensures the transpose operation is an $O(N)$ operation based on the Rank $N$ of the tensor, not the total size $L$ of the data. 

---

## Tensor Printing (`<<` Operator)

Until you can see the structured output, debugging any complex operation (like transpose) is difficult. Overloading the stream insertion operator is a must-have utility.

### Implementation Principle

Since your tensor is flat, you must implement a recursive helper function to use the **shape** and **strides** to correctly insert brackets and separate elements.

| Rank $N$ | Goal | Structure |
| :---: | :--- | :--- |
| 1 (Vector) | Print elements separated by commas. | `[e0, e1, e2]` |
| 2 (Matrix) | Print rows separated by commas, enclosed in outer brackets. | `[[e00, e01], [e10, e11]]` |
| $N$ (General) | Print nested structures with $N$ levels of brackets. | `[ [ [ ... ] ] ]` |

### Mathematical and Logic Formalization

The logic involves iterating through the flat array index $k$ from $0$ to $L-1$, but checking the coordinates at each step to determine if a bracket or newline is needed.

The core idea is to find the point $k$ where the coordinate $i_{N-1}$ (the fastest-changing index) resets to $0$.

$$
\text{Start New Row/Slice } \iff \quad k \neq 0 \quad \text{AND} \quad \text{The fastest-changing coordinate resets to } 0
$$

The fastest index resets when $k$ is a multiple of the last stride $s_{N-1}$ (which is 1), but the *second-to-last* index $i_{N-2}$ resets when $k$ is a multiple of the stride $s_{N-2}$.

The logic requires calculating the coordinates $\mathbf{I}=(i_0, \dots, i_{N-1})$ for the current flat index $k$ and printing an opening/closing bracket whenever a coordinate $i_j$ changes from its maximum value $d_j-1$ back to $0$.

### 1. Goal: Overload the Stream Insertion Operator

You will define a non-member function that overloads the stream insertion operator (`<<`). This function allows users to print your tensor like any standard C++ object:

$$\text{std::cout} \ll \mathbf{A} \ll \text{std::endl};$$

### Formal Signature
The function must take a reference to the output stream and a reference to your tensor:

$$\mathcal{O} = \text{operator}\ll (\mathcal{O}, \mathbf{A})$$

### 2. Core Printing Strategy: The Recursive Helper

To handle arbitrary ranks, the main `operator<<` function should call a **private, recursive helper function** (let's call it `print_recursive`) that manages the nested structure based on the tensor's rank.

#### Helper Function Parameters

The `print_recursive` function needs three things to track its progress:

1.  **`os` (Output Stream):** Where to print the output.
2.  **`current_dim` (Current Rank Level):** An integer that tracks which dimension you are currently iterating over (e.g., 0 for the outermost dimension, $N-1$ for the innermost).
3.  **`flat_index` (Starting Data Index):** A reference to an integer that tracks the current position in the tensor's flat data array. This must be a **reference** because all recursive calls must modify and share the same global index pointer.

### 3. Recursive Logic

The logic is based on traversing the dimensions from the **outermost** ($current\_dim = 0$) to the **innermost** ($current\_dim = N-1$).

#### A. Base Case: Innermost Dimension (Vector)

The recursion stops when you reach the last dimension: $current\_dim = N-1$.

1.  Print an opening bracket: `[`
2.  **Loop** $d_{N-1}$ times (where $d_{N-1}$ is the size of the last dimension):
    * Print the element: $\mathbf{A}.\text{data}[\text{flat\_index}]$
    * Increment the flat index: $\text{flat\_index} \leftarrow \text{flat\_index} + 1$
    * If it is **not** the last element of the row, print a comma and space: `, `
3.  Print a closing bracket: `]`
4.  **Return** from recursion.

#### B. Recursive Step: Outer Dimensions ($current\_dim < N-1$)

For any dimension that is not the innermost:

1.  Print an opening bracket: `[`
2.  **Loop** $d_{current\_dim}$ times (where $d_{current\_dim}$ is the size of the current dimension):
    * **Recursive Call:** Call `print_recursive(os, current_dim + 1, flat_index)`. This processes the next, nested dimension.
    * If it is **not** the last slice/row/sub-tensor in this dimension, print a comma and a newline (or space): `, \n`
3.  Print a closing bracket: `]`
4.  **Return** from recursion.

### 4. Initialization and Call Flow

The main `operator<<` function sets up the call:

1.  Initialize a tracking index: $\text{flat\_index} = 0$.
2.  Determine the total rank: $N = |\mathbf{A}.\text{shape}|$.
3.  Call the helper: `print_recursive(os, 0, flat_index)`.

This recursive structure ensures the output will be properly nested, correctly using the shape vector to determine how many elements belong in each nested "row" or "slice" before a closing bracket is placed.
