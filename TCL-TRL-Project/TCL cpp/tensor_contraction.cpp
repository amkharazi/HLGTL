#include <torch/extension.h>

torch::Tensor tensor_contraction(const torch::Tensor& T, 
                                  const torch::Tensor& A, 
                                  const torch::Tensor& B, 
                                  const torch::Tensor& C) {
    // Check that the input tensors are on CPU and of the correct shape
    TORCH_CHECK(T.dim() == 4, "Input tensor T should be 4D");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2, "A, B, C should be 2D");

    // Extract dimensions
    int64_t B_size = T.size(0);  // Batch size
    int64_t R1 = A.size(0);
    int64_t R2 = B.size(0);
    int64_t R3 = C.size(0);
    int64_t C_size = T.size(1);
    int64_t H = T.size(2);
    int64_t W = T.size(3);

    // Prepare output tensor
    auto result = torch::zeros({B_size, R1, R2, R3}, T.options());

    // Perform the mode products
    for (int64_t b = 0; b < B_size; ++b) {
        for (int64_t r1 = 0; r1 < R1; ++r1) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    // Each element in result is computed using A, B, C
                    for (int64_t c = 0; c < C_size; ++c) {
                        result[b][r1][h][w] += A[r1][c] * T[b][c][h][w];
                    }
                }
            }
        }
    }

    return result;
}

PYBIND11_MODULE(tensor_contraction_cpp, m) {
    m.def("tensor_contraction", &tensor_contraction, "Tensor contraction operation");
}
