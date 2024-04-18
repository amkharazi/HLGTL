import torch
import torch.nn as nn
class reshape(nn.Module):
    def __init__(self, split, map_type=1, device='cpu'):
        super(reshape, self).__init__()
        
        self.split = split
        self.map_type = map_type
        self.device = device
        
    def split_into_chunks(self, tensor):
   
        chunks = []
        _, C, H, W = tensor.shape
        
        if self.map_type == 1:
            # Approach 1 : Attached
            C_indices, H_indices, W_indices = [
                [sum(dim // self.split[i] for _ in range(j)) for j in range(self.split[i] + 1)]
                for i, dim in enumerate([C, H, W])
            ]
            
            for i in range(self.split[0]):
                for j in range(self.split[1]):
                    for k in range(self.split[2]):
                        chunk = tensor[:, 
                                       C_indices[i]:C_indices[i+1], 
                                       H_indices[j]:H_indices[j+1], 
                                       W_indices[k]:W_indices[k+1]]
                        chunks.append(chunk)
        
        elif self.map_type == 2:
            # Approach 2 : Compressed
            for i in range(self.split[0]):
                for j in range(self.split[1]):
                    for k in range(self.split[2]):
                        C_stride_indices = torch.arange(i, C, self.split[0]).to(self.device)
                        H_stride_indices = torch.arange(j, H, self.split[1]).to(self.device)
                        W_stride_indices = torch.arange(k, W, self.split[2]).to(self.device)

                        chunk = tensor.index_select(1, C_stride_indices)
                        chunk = chunk.index_select(2, H_stride_indices)
                        chunk = chunk.index_select(3, W_stride_indices)
                        chunks.append(chunk)
        else:
            raise(ValueError('Wrong map type (choose 1 or 2)'))

        return chunks
    
    def stack_chunks_to_form_tensor(self, chunks, batch_size):
        
        if len(chunks) != self.split[0] * self.split[1] * self.split[2]:
            raise(RuntimeError('Number of chunks does not match the expected number based on split size.'))
   
        result = torch.reshape(torch.stack(chunks), 
                               (batch_size, self.split[0], self.split[1], self.split[2], 
                                *chunks[0].shape[1:]))
        return result

    def forward(self, x):
        batch_size = x.shape[0]
        chunks = self.split_into_chunks(x)
        output = self.stack_chunks_to_form_tensor(chunks, batch_size)
        return output