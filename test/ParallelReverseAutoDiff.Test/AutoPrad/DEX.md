# Dex - Deferred Execution for PRAD

Dex is a deferred execution layer built on top of the Parallel Reverse Auto Differentiation (PRAD) library. It allows you to build computation graphs that are executed later, enabling better optimization and more flexible gradient computation.

## Basic Usage

```csharp
// Create a new Dex instance
var dex = new Dex();

// Create some tensors
var t1 = new Tensor(new[] { 2, 2 }, new double[] { 1, 2, 3, 4 });
var t2 = new Tensor(new[] { 2, 2 }, new double[] { 5, 6, 7, 8 });

// Build computation using fluent interface
var result = dex.Seed(t1)
   .Then(x => x * t2)
   .Then(x => x + t3)
   .Forward();

// Access gradients
var gradient = dex.Gradients[t1];
Alternative Syntax
You can also use the DoOp method for more complex expressions:
csharpvar exp = dex.DoOp(() => t1 * t2 + t3);
var result = dex.Forward();
exp.BackAccumulate();
```

Supported Operations

Addition (+)
Subtraction (-)
Multiplication (*)
Division (/)
Matrix Multiplication (MatMul)

Features

Deferred execution of tensor operations
Automatic operation reordering for optimal gradient computation
Smart branching for reused expressions
Integration with PRAD's gradient computation
Support for both accumulating and replacing gradients

Gradient Computation
```csharp
// Accumulate gradients
exp.BackAccumulate();

// Replace gradients instead of accumulating
exp.BackReplace();
```

Benefits

Optimization: Operations can be reordered and optimized before execution
Flexibility: Build complex computations that can be executed multiple times
Clarity: Clean, fluent interface for building computation graphs
Integration: Seamless integration with PRAD's existing functionality

Dependencies

PRAD Library (Parallel Reverse Auto Differentiation)

Examples
Complex expression with multiple reuse:
```csharp
var exp = dex.DoOp(() => 
    (t1 * t2 + t1 * t3) / (t2 + t3));
var result = dex.Forward();
exp.BackAccumulate();
Matrix operations:
csharpvar result = dex.Seed(matrix1)
    .Then(x => x.MatMul(matrix2))
    .Then(x => x + bias)
    .Forward();
```

Notes

Operations are not executed until Forward() is called
Gradients are computed when calling BackAccumulate() or BackReplace()
The system automatically handles branching for reused expressions