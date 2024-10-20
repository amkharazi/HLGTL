// tcl2.cpp

#include </home/bady/.local/lib/python3.10/site-packages/torch/include/torch/extension.h>
#include <vector>

// Declarations of CUDA functions
std::vector<torch::Tensor> tcl2_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor w3);

std::vector<torch::Tensor> tcl2_backward_cuda(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor w3);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> tcl2_forward(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor w3)
{
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
    CHECK_INPUT(w3);
    return tcl2_forward_cuda(x, w1, w2, w3);
}

std::vector<torch::Tensor> tcl2_backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor w3)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
    CHECK_INPUT(w3);
    return tcl2_backward_cuda(grad_y, x, w1, w2, w3);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tcl2_forward, "TCL2 forward (C++)");
    m.def("backward", &tcl2_backward, "TCL2 backward (C++)");
}
