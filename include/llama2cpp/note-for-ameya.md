### Week of 03/18/2024
 - refactored tokenizer
 - started refactoring the Transformer.

### TODO:
 - Once the model pieces are together, separate them into individual layers still as pointers.
 - Once that is done, refactor them into tensors.
 - No need to memory map, copy them into buffers in the memory as that would be more scalable for cuda in future.
 - Redo SWIGLU
 - Implement tensor slicing