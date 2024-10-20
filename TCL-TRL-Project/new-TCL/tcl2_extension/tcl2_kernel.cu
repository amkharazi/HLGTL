// tcl2_kernel.cu

#include <torch/extension.h>
// #include </home/bady/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include/torch/cuda.h>
// #include </home/bady/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/include/cuda_runtime.h>
// #include <vector>

// // Forward CUDA kernel
// __global__ void tcl2_forward_kernel(
//     const float* __restrict__ x,
//     const float* __restrict__ w1,
//     const float* __restrict__ w2,
//     const float* __restrict__ w3,
//     float* __restrict__ y,
//     int batch_size,
//     int input_dim,
//     int rank1,
//     int rank2,
//     int rank3,
//     int output_dim)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= batch_size * output_dim) return;

//     int b = idx / output_dim;
//     int o = idx % output_dim;

//     // Decode o into indices for w1, w2, w3
//     int size_w3 = rank3;
//     int size_w2 = rank2;
//     int size_w1 = rank1;

//     int c = o % size_w3;
//     int b_idx = (o / size_w3) % size_w2;
//     int a = (o / (size_w3 * size_w2)) % size_w1;

//     // Compute the corresponding indices in w1, w2, w3
//     // Assuming W = kron(kron(w1, w2), w3)
//     // So, W has dimensions (input_dim, output_dim)
//     // input_dim = dim1 * dim2 * dim3
//     // output_dim = rank1 * rank2 * rank3

//     // Compute the contribution from x and W
//     float sum = 0.0;
//     for (int i = 0; i < input_dim; ++i)
//     {
//         // Decode i into indices for input_shape (assuming input_shape is (dim1, dim2, dim3))
//         // For simplicity, assume input_shape is (64,8,8) as per your example
//         int dim3 = 8;
//         int dim2 = 8;
//         int dim1 = 64;

//         int c_in = i % dim3;
//         int b_in = (i / dim3) % dim2;
//         int a_in = (i / (dim3 * dim2)) % dim1;

//         // Kronecker product indexing
//         // W_{i,o} = w1[a_in, a] * w2[b_in, b_idx] * w3[c_in, c]

//         float w1_val = w1[a_in * size_w1 + a];
//         float w2_val = w2[b_in * size_w2 + b_idx];
//         float w3_val = w3[c_in * size_w3 + c];

//         float W_val = w1_val * w2_val * w3_val;

//         sum += x[b * input_dim + i] * W_val;
//     }

//     y[b * output_dim + o] = sum;
// }

// // Backward CUDA kernel
// __global__ void tcl2_backward_kernel(
//     const float* __restrict__ grad_y,
//     const float* __restrict__ x,
//     float* __restrict__ grad_w1,
//     float* __restrict__ grad_w2,
//     float* __restrict__ grad_w3,
//     int batch_size,
//     int input_dim,
//     int rank1,
//     int rank2,
//     int rank3,
//     int output_dim)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= (rank1 * rank2 * rank3)) return;

//     int a = idx / (rank2 * rank3);
//     int b_idx = (idx / rank3) % rank2;
//     int c = idx % rank3;

//     float grad_w1_val = 0.0f;
//     float grad_w2_val = 0.0f;
//     float grad_w3_val = 0.0f;

//     for (int b = 0; b < batch_size; ++b)
//     {
//         for (int i = 0; i < input_dim; ++i)
//         {
//             // Decode i into indices for input_shape (assuming input_shape is (dim1, dim2, dim3))
//             int dim3 = 8;
//             int dim2 = 8;
//             int dim1 = 64;

//             int c_in = i % dim3;
//             int b_in = (i / dim3) % dim2;
//             int a_in = (i / (dim3 * dim2)) % dim1;

//             if (a_in == a)
//             {
//                 // W_{i,o} = w1[a_in, a] * w2[b_in, b_idx] * w3[c_in, c]
//                 float x_val = x[b * input_dim + i];
//                 grad_w1_val += grad_y[b * output_dim + (a * rank2 * rank3 + b_idx * rank3 + c)] * x_val * w2[b_in * rank2 + b_idx] * w3[c_in * rank3 + c];
//             }

//             if (b_in == b_idx)
//             {
//                 // W_{i,o} = w1[a_in, a] * w2[b_in, b_idx] * w3[c_in, c]
//                 float x_val = x[b * input_dim + i];
//                 grad_w2_val += grad_y[b * output_dim + (a * rank2 * rank3 + b_idx * rank3 + c)] * x_val * w1[a_in * rank1 + a] * w3[c_in * rank3 + c];
//             }

//             if (c_in == c)
//             {
//                 // W_{i,o} = w1[a_in, a] * w2[b_in, b_idx] * w3[c_in, c]
//                 float x_val = x[b * input_dim + i];
//                 grad_w3_val += grad_y[b * output_dim + (a * rank2 * rank3 + b_idx * rank3 + c)] * x_val * w1[a_in * rank1 + a] * w2[b_in * rank2 + b_idx];
//             }
//         }
//     }

//     // Atomic operations to avoid race conditions
//     atomicAdd(&grad_w1[a * rank1 + a], grad_w1_val);
//     atomicAdd(&grad_w2[b_idx * rank2 + b_idx], grad_w2_val);
//     atomicAdd(&grad_w3[c * rank3 + c], grad_w3_val);
// }

// // C++ interface
// std::vector<torch::Tensor> tcl2_forward_cuda(
//     torch::Tensor x,
//     torch::Tensor w1,
//     torch::Tensor w2,
//     torch::Tensor w3)
// {
//     // Get dimensions
//     int batch_size = x.size(0);
//     int input_dim = x.size(1);
//     int rank1 = w1.size(1);
//     int rank2 = w2.size(1);
//     int rank3 = w3.size(1);
//     int output_dim = rank1 * rank2 * rank3;

//     // Allocate output
//     auto y = torch::zeros({batch_size, output_dim}, x.options());

//     // Define CUDA kernel dimensions
//     int threads = 256;
//     int blocks = (batch_size * output_dim + threads - 1) / threads;

//     // Launch forward kernel
//     tcl2_forward_kernel<<<blocks, threads>>>(
//         x.data_ptr<float>(),
//         w1.data_ptr<float>(),
//         w2.data_ptr<float>(),
//         w3.data_ptr<float>(),
//         y.data_ptr<float>(),
//         batch_size,
//         input_dim,
//         rank1,
//         rank2,
//         rank3,
//         output_dim
//     );

//     // Wait for GPU to finish
//     cudaDeviceSynchronize();

//     return {y};
// }

// std::vector<torch::Tensor> tcl2_backward_cuda(
//     torch::Tensor grad_y,
//     torch::Tensor x,
//     torch::Tensor w1,
//     torch::Tensor w2,
//     torch::Tensor w3)
// {
//     // Get dimensions
//     int batch_size = x.size(0);
//     int input_dim = x.size(1);
//     int rank1 = w1.size(1);
//     int rank2 = w2.size(1);
//     int rank3 = w3.size(1);
//     int output_dim = rank1 * rank2 * rank3;

//     // Allocate gradients
//     auto grad_w1 = torch::zeros_like(w1);
//     auto grad_w2 = torch::zeros_like(w2);
//     auto grad_w3 = torch::zeros_like(w3);

//     // Define CUDA kernel dimensions
//     int threads = 256;
//     int blocks = (rank1 * rank2 * rank3 + threads - 1) / threads;

//     // Launch backward kernel
//     tcl2_backward_kernel<<<blocks, threads>>>(
//         grad_y.data_ptr<float>(),
//         x.data_ptr<float>(),
//         grad_w1.data_ptr<float>(),
//         grad_w2.data_ptr<float>(),
//         grad_w3.data_ptr<float>(),
//         batch_size,
//         input_dim,
//         rank1,
//         rank2,
//         rank3,
//         output_dim
//     );

//     // Wait for GPU to finish
//     cudaDeviceSynchronize();

//     return {grad_w1, grad_w2, grad_w3};
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward", &tcl2_forward_cuda, "TCL2 forward (CUDA)");
//     m.def("backward", &tcl2_backward_cuda, "TCL2 backward (CUDA)");
// }
