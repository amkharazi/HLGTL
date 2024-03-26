import torch

def topk_accuracy(output, target, k):
    _, topk_indices = torch.topk(input=output, k=k, dim=1, largest=True, sorted=True)
    correct = topk_indices.eq(target.view(-1, 1).expand_as(topk_indices)).sum()
    return (correct / output.shape[0]) * 100.0, correct, output.shape[0]
