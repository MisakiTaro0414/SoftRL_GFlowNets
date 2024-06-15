import torch

def apply_NM_sparsity(tensor: torch.Tensor, s: float, N: int, M: int) -> torch.Tensor:


    # total number of elements in the tensor
    total_elements = tensor.numel()

    # effective sparsity of the tensor
    effective_sparsity = min(s, N / M)

    # total prunable elements
    total_elements_to_prune = int(effective_sparsity * total_elements)

    # reshape the shape of the tensor to 2D tensor with M columns
    original_shape = tensor.shape
    last_dim = original_shape[-1]
    num_full_blocks = last_dim // M
    effective_last_dim = num_full_blocks * M
    reshaped_tensor = tensor[..., :effective_last_dim].reshape(-1, M)

    # compute l1 norm of the reshaped tensor and sort the indices
    abs_values = reshaped_tensor.abs()
    flat_abs_values = abs_values.flatten()
    _, global_indices = flat_abs_values.sort()

    # create a mask to prune the tensor and initialize the pruned elements count
    mask = torch.ones_like(reshaped_tensor, dtype=torch.bool)
    pruned_elements_count = 0

    # iterate through the sorted indices to allocate prunings
    for idx in global_indices:
        if pruned_elements_count >= total_elements_to_prune:
            break

        block_index = idx // M
        in_block_index = idx % M

        # we don't prune more than N elements per block
        if mask[block_index, :].sum() > (M - N):  
            mask[block_index, in_block_index] = False
            pruned_elements_count += 1

    # apply the mask to the reshaped tensor
    reshaped_tensor *= mask

    # reshape the tensor back to its original shape
    sparsified_tensor = torch.cat([
        reshaped_tensor.reshape(original_shape[:-1] + (effective_last_dim,)),
        tensor[..., effective_last_dim:]  # Remaining elements untouched
    ], dim=-1)

    return sparsified_tensor

# Example usage
tensor = torch.randn(3,5, 5, 24)  # Example 2D tensor
sparsified_tensor = apply_NM_sparsity(tensor, s=0.7, N=4, M=8)
print("Sparsified degree: ", (sparsified_tensor == 0).float().mean().item())

## SOURCES ##
# Google search for "NP sparsity tensor pruning"
# pytorch.org documentation

# NOTE:
# This implementation is a simple example and may not be the most efficient or optimized solution
# since it is using a for loop to iterate through the sorted indices. The vectorized implementation is 
# not ready yet.






