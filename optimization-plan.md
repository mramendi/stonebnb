[x] Ensure no unquantized weight tensors (*not even slices*) are attached to autograd graph
    - Replaced F.linear() with torch.matmul() in MoELinear4Bit forward pass
    - F.linear creates MmBackward0 nodes that save dequantized weights
    - torch.matmul in custom autograd Function lets us control what gets saved
    - Only quantized weights saved to ctx (just a reference, minimal overhead)
    - Test: Run 10 training steps on old branch vs new, compare loss values (should be identical)
[ ] Implement Virtual3DTensor as the only way to access dequantized expert weight tensors. Ah that stage the source is actual 3d Tensors in a 1-deep LRU cache (so two sequential layer[i] requests *within the same layer* take one dequant)
[ ] Change the stonebnb format to store each expert layer as 72 (or however many there are) 2D tensors. In Virtual3DTensor, dequant only the requested 2D tensor just-in-time
[ ] Implement an LRU cache of 2D tensors accessed by [layer_id,i], with a tunable limit stored in memory (right down to only one 2D tensor for maximum memory saving). Dequant 2D tensor on cache miss
[ ] Implement CPU offloaded mode where the standard BF16 model is the source, no quants are involved, MoE layers live on CPU and the LRU cache pulls them onto GPU on cache miss
[ ] STRETCH GOAL: Implement dynamic sizing of the LRU cache (to enable more caching on shorter batches)


