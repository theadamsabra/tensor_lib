# tensor_lib

A C++ based Tensor library. The purpose of this repository is for me to learn C++ while finally testing myself on quite low level operations that I regularly take for granted in similar libraries like PyTorch, NumPy, JAX, and Tensorflow.

If you think this will ever be production-grade, I have AGI to sell you.

## Tensor Development Roadmap

### Phase 1: Basic Operations & Utility (Current Focus)

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Stream Insertion Operator** | Implement `operator<<` for structured tensor printing. | **TODO (Next)** |
| **Move Semantics** | Implement Move Constructor and Move Assignment for efficiency. | TO DO |
| **Matrix Multiplication** | Implement the general $\mathbf{A} \times \mathbf{B}$ matrix product. | TO DO |

***

### Phase 2: Creation & Generation

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Tensor Creation: Zeros** | Static factory method to create a tensor filled with $0$. | TO DO |
| **Tensor Creation: Ones** | Static factory method to create a tensor filled with $1$. | TO DO |
| **Tensor Creation: Random** | Static factory method to create a tensor with random values. | TO DO |

***

### Phase 3: Advanced Indexing & Reshaping

| Feature | Description | Status |
| :--- | :--- | :--- |
| **Reshape** | Change the tensor's dimensions without changing the data or size (e.g., $2 \times 3 \to 6 \times 1$). | TO DO |
| **Slicing** | Implement the ability to select a sub-section of the tensor (e.g., getting rows 1-3 of a matrix). | TO DO |

***

### Phase 4: Package Usability & Distribution

| Feature | Description | Status |
| :--- | :--- | :--- |
| **CMake Build System** | Write the necessary `CMakeLists.txt` files to allow users to build and integrate the library easily. | TO DO |
| **Installation Target** | Define installation steps so users can run `make install` to put the library headers in a standard location. | TO DO |
