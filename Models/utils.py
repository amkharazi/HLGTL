def topk(output, target, k):
    correct = 0.0
    batch_size = output.size(0)
    _, topk_indices = output.topk(k, dim=1, largest=True, sorted=True)
    target = target.view(-1, 1).expand_as(topk_indices)

    correct += (topk_indices == target).sum().item()

    return (correct / batch_size) * 100.0
