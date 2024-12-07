# Bugs and Improvements

## Memory - Active
### Excessive NP Loading
Instead of passing the images as numpy array to the dataset, pass the paths instead

### CUDA Memory
It has been difficult to fit the tensors into the gpu on the robotics vm. To debug if 
statements are needed at various points using the following code:
```python
if torch.cuda.is_available():
    print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
    print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
```

### Manage Copies
Each of the transformations is meant to make a copy of the image, fix this or del as 
soon as possible to manage memory constraints


## Training Scripts
TODO: Set up argument parser for command-line flags for plotting and using previous weights
TODO: Consider adding confidence as output, this would also have an associated loss
TODO: Verify seed does not affect randomization used in the model and processing