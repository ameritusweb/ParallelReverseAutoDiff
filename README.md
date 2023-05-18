# ParallelReverseAutoDiff
Parallel Reverse Mode Automatic Differentiation in C#

[![NuGet version (parallelreverseautodiff)](https://img.shields.io/nuget/v/parallelreverseautodiff?style=flat-square)](https://www.nuget.org/packages/parallelreverseautodiff/)
![Nuget](https://img.shields.io/nuget/dt/parallelreverseautodiff?style=flat-square)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7f9f69794dd74a97aeaac17ebd1580ec)](https://app.codacy.com/gh/ameritusweb/ParallelReverseAutoDiff/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

ParallelAutoDiff is a thread-safe C# library for reverse mode automatic differentiation, optimized for parallel computation. It leverages semaphores and locks to coordinate between threads, ensuring accuracy during gradient accumulation. Each operation in the library is implemented as a node with a forward and a backward function, facilitating efficient calculation of derivatives. A unique aspect of this library is its use of the visitor pattern: it includes a specialized 'Neural Network Visitor' which traverses neural network nodes across different threads. This visitor is responsible for gradient accumulation on nodes shared across multiple threads. This design allows for parallelized computations while maintaining consistency and avoiding race conditions. The result is an efficient, scalable automatic differentiation solution, ideal for machine learning applications and neural network training.

[API Documentation](https://ameritusweb.github.io/ParallelReverseAutoDiff/api/index.html)

## Prerequisites
Download and install the [Cuda Toolkit 12.0](https://developer.nvidia.com/cuda-12-0-0-download-archive) if you want to use the CudaMatrixMultiplyOperation.

## Supported Operations

### Regular Operations
AmplifiedSigmoidOperation - Used for gradient amplification.

ApplyDropoutOperation

BatchNormalizationOperation

CudaMatrixMultiplyOperation - Leverages the GPU for fast computation.

GELUOperation

HadamardProductOperation

LayerNormalizationOperation

LeakyReLUOperation

MatrixAddOperation

MatrixAddThreeOperation

MatrixMultiplyOperation

MatrixMultiplyScalarOperation

MatrixTransposeOperation

ReLUOperation

ScaleAndShiftOperation

SigmoidOperation

SoftmaxOperation

StretchedSigmoidOperation

SwigLUOperation

SwishOperation

TanhOperation

### Deep Operations
These types of operations operate on instances of the DeepMatrix class which is a 3-D matrix.
The first dimension is the channel size and the second and third dimensions are the row and column sizes respectively.

DeepBatchNormalizationOperation

DeepConvolutionOperation

DeepLeakyReLUOperation

DeepMaxPoolOperation

DeepReLUOperation

DeepScaleAndShiftOperation

FlattenOperation

## Usage

### Create an architecture JSON file

Here is an example:
```json
{
  "timeSteps": [
    {
      "startOperations": [
        {
          "id": "projectedInput",
          "description": "Multiply the input with the weight matrix",
          "type": "MatrixMultiplyOperation",
          "inputs": [ "We", "inputSequence[t]" ],
          "gradientResultTo": [ "dWe", null ]
        },
        {
          "id": "embeddedInput",
          "description": "Add the bias",
          "type": "MatrixAddOperation",
          "inputs": [ "projectedInput", "be" ],
          "gradientResultTo": [ null, "dbe" ]
        }
      ],
      "layers": [
        {
          "operations": [
            {
              "id": "wf_currentInput",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Wf[layerIndex]", "currentInput" ],
              "gradientResultTo": [ "dWf[layerIndex]", null ]
            },
            {
              "id": "uf_previousHiddenState",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Uf[layerIndex]", "previousHiddenState" ],
              "gradientResultTo": [ "dUf[layerIndex]", null ]
            },
            {
              "id": "f_add",
              "type": "MatrixAddThreeOperation",
              "inputs": [ "wf_currentInput", "uf_previousHiddenState", "bf[layerIndex]" ],
              "gradientResultTo": [ null, null, "dbf[layerIndex]" ]
            },
            {
              "id": "intermediate_f_1",
              "description": "Compute the forget gate",
              "type": "MatrixTransposeOperation",
              "inputs": [ "f_add" ]
            },
            {
              "id": "intermediate_f_2",
              "description": "Compute the forget gate",
              "type": "LayerNormalizationOperation",
              "inputs": [ "intermediate_f_1" ]
            },
            {
              "id": "intermediate_f_3",
              "description": "Compute the forget gate",
              "type": "MatrixTransposeOperation",
              "inputs": [ "intermediate_f_2" ]
            },
            {
              "id": "f",
              "description": "Compute the forget gate",
              "type": "AmplifiedSigmoidOperation",
              "inputs": [ "intermediate_f_3" ],
              "setResultTo": "f[t][layerIndex]"
            },
            {
              "id": "wi_currentInput",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Wi[layerIndex]", "currentInput" ],
              "gradientResultTo": [ "dWi[layerIndex]", null ]
            },
            {
              "id": "ui_previousHiddenState",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Ui[layerIndex]", "previousHiddenState" ],
              "gradientResultTo": [ "dUi[layerIndex]", null ]
            },
            {
              "id": "i_add",
              "type": "MatrixAddThreeOperation",
              "inputs": [ "wi_currentInput", "ui_previousHiddenState", "bi[layerIndex]" ],
              "gradientResultTo": [ null, null, "dbi[layerIndex]" ]
            },
            {
              "id": "intermediate_i_1",
              "description": "Compute the input gate",
              "type": "MatrixTransposeOperation",
              "inputs": [ "i_add" ]
            },
            {
              "id": "intermediate_i_2",
              "description": "Compute the input gate",
              "type": "LayerNormalizationOperation",
              "inputs": [ "intermediate_i_1" ]
            },
            {
              "id": "intermediate_i_3",
              "description": "Compute the input gate",
              "type": "MatrixTransposeOperation",
              "inputs": [ "intermediate_i_2" ]
            },
            {
              "id": "i",
              "description": "Compute the input gate",
              "type": "AmplifiedSigmoidOperation",
              "inputs": [ "intermediate_i_3" ],
              "setResultTo": "i[t][layerIndex]"
            },
            {
              "id": "wc_currentInput",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Wc[layerIndex]", "currentInput" ],
              "gradientResultTo": [ "dWc[layerIndex]", null ]
            },
            {
              "id": "uc_previousHiddenState",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Uc[layerIndex]", "previousHiddenState" ],
              "gradientResultTo": [ "dUc[layerIndex]", null ]
            },
            {
              "id": "cHat_add",
              "type": "MatrixAddThreeOperation",
              "inputs": [ "wc_currentInput", "uc_previousHiddenState", "bc[layerIndex]" ],
              "gradientResultTo": [ null, null, "dbc[layerIndex]" ]
            },
            {
              "id": "intermediate_cHat_1",
              "description": "Compute the candidate memory cell state",
              "type": "MatrixTransposeOperation",
              "inputs": [ "cHat_add" ]
            },
            {
              "id": "intermediate_cHat_2",
              "description": "Compute the candidate memory cell state",
              "type": "LayerNormalizationOperation",
              "inputs": [ "intermediate_cHat_1" ]
            },
            {
              "id": "intermediate_cHat_3",
              "description": "Compute the candidate memory cell state",
              "type": "MatrixTransposeOperation",
              "inputs": [ "intermediate_cHat_2" ]
            },
            {
              "id": "cHat",
              "description": "Compute the candidate memory cell state",
              "type": "TanhOperation",
              "inputs": [ "intermediate_cHat_3" ],
              "setResultTo": "cHat[t][layerIndex]"
            },
            {
              "id": "f_previousMemoryCellState",
              "type": "HadamardProductOperation",
              "inputs": [ "f[t][layerIndex]", "previousMemoryCellState" ]
            },
            {
              "id": "i_cHat",
              "type": "HadamardProductOperation",
              "inputs": [ "i[t][layerIndex]", "cHat[t][layerIndex]" ]
            },
            {
              "id": "newC",
              "description": "Compute the memory cell state",
              "type": "MatrixAddOperation",
              "inputs": [ "f_previousMemoryCellState", "i_cHat" ]
            },
            {
              "id": "newCTransposed",
              "type": "MatrixTransposeOperation",
              "inputs": [ "newC" ]
            },
            {
              "id": "newCNormalized",
              "type": "LayerNormalizationOperation",
              "inputs": [ "newCTransposed" ]
            },
            {
              "id": "c",
              "type": "MatrixTransposeOperation",
              "inputs": [ "newCNormalized" ],
              "setResultTo": "c[t][layerIndex]"
            },
            {
              "id": "wo_currentInput",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Wo[layerIndex]", "currentInput" ],
              "gradientResultTo": [ "dWo[layerIndex]", null ]
            },
            {
              "id": "uo_previousHiddenState",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Uo[layerIndex]", "previousHiddenState" ],
              "gradientResultTo": [ "dUo[layerIndex]", null ]
            },
            {
              "id": "o_add",
              "type": "MatrixAddThreeOperation",
              "inputs": [ "wo_currentInput", "uo_previousHiddenState", "bo[layerIndex]" ],
              "gradientResultTo": [ null, null, "dbo[layerIndex]" ]
            },
            {
              "id": "o",
              "description": "Compute the output gate",
              "type": "LeakyReLUOperation",
              "inputs": [ "o_add" ],
              "setResultTo": "o[t][layerIndex]"
            },
            {
              "id": "c_tanh",
              "type": "TanhOperation",
              "inputs": [ "c" ]
            },
            {
              "id": "newH",
              "type": "HadamardProductOperation",
              "inputs": [ "o[t][layerIndex]", "c_tanh" ]
            },
            {
              "id": "keys",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Wk[layerIndex]", "embeddedInput" ],
              "gradientResultTo": [ "dWk[layerIndex]", null ]
            },
            {
              "id": "queries",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Wq[layerIndex]", "previousHiddenState" ],
              "gradientResultTo": [ "dWq[layerIndex]", null ]
            },
            {
              "id": "values",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "Wv[layerIndex]", "embeddedInput" ],
              "gradientResultTo": [ "dWv[layerIndex]", null ]
            },
            {
              "id": "queriesTranspose",
              "type": "MatrixTransposeOperation",
              "inputs": [ "queries" ]
            },
            {
              "id": "dotProduct",
              "description": "Compute the dot product of the queries and keys",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "keys", "queriesTranspose" ]
            },
            {
              "id": "scaledDotProduct",
              "description": "Scale the dot product",
              "type": "MatrixMultiplyScalarOperation",
              "inputs": [ "dotProduct", "scaledDotProductScalar" ]
            },
            {
              "id": "scaledDotProductTranspose",
              "type": "MatrixTransposeOperation",
              "inputs": [ "scaledDotProduct" ]
            },
            {
              "id": "attentionWeights",
              "type": "SoftmaxOperation",
              "inputs": [ "scaledDotProductTranspose" ]
            },
            {
              "id": "attentionOutput",
              "type": "MatrixMultiplyOperation",
              "inputs": [ "attentionWeights", "values" ]
            },
            {
              "id": "newHWithAttentionOutput",
              "type": "MatrixAddOperation",
              "inputs": [ "newH", "attentionOutput" ]
            },
            {
              "id": "newHWithAttentionOutputTranspose",
              "type": "MatrixTransposeOperation",
              "inputs": [ "newHWithAttentionOutput" ]
            },
            {
              "id": "normalizedNewH",
              "type": "LayerNormalizationOperation",
              "inputs": [ "newHWithAttentionOutputTranspose" ]
            },
            {
              "id": "h",
              "type": "MatrixTransposeOperation",
              "inputs": [ "normalizedNewH" ],
              "setResultTo": "h[t][layerIndex]"
            }
          ]
        }
      ],
      "endOperations": [
        {
          "id": "v_h",
          "type": "MatrixMultiplyOperation",
          "inputs": [ "V", "hFromCurrentTimeStepAndLastLayer" ],
          "gradientResultTo": [ "dV", null ]
        },
        {
          "id": "v_h_b",
          "type": "MatrixAddOperation",
          "inputs": [ "v_h", "b" ],
          "gradientResultTo": [ null, "db" ]
        },
        {
          "id": "output_t",
          "type": "AmplifiedSigmoidOperation",
          "inputs": [ "v_h_b" ],
          "setResultTo": "output[t]"
        }
      ]
    }
  ]
}
```

Each operation in the JSON represents a step in a computational graph used for automatic differentiation. Here's what each field means:

* "timeSteps": This is an array that represents the sequence of computational operations. Each element in the array is an object that corresponds to a computational timestep.

* "startOperations": This is an array that defines the initial operations for the current timestep.

* "layers": This represents a sequence of operations corresponding to the layers of the network. Each operation in a layer is a step in the computation, and the order of operations matters, as some operations depend on the results of previous operations. 

* "endOperations": This is an array that defines the final operations for the current timestep.

Each operation object in "startOperations", "layers", or "endOperations" has several fields:

* "id": This is a unique identifier for the operation.

* "description": This is a human-readable description of what the operation does.

* "type": This specifies the type of the operation.

* "inputs": This is an array that lists the inputs for the operation. These are the identifiers of other nodes in the computational graph. The identifiers are either defined in the computational graph, or in operation finders declared in code when building an instance of the ComputationGraph class, for example the SelfAttentionMultiLayerLSTMComputationGraph class which is a subclass of ComputationGraph.

* "gradientResultTo": This is an array that specifies where the results of the backward pass (i.e., the computed gradients) should be stored. A null value means that the gradient with respect to the input is not stored. There is an implicit mapping between the gradient and the input based on its position in the array.

* "setResultTo": This is used to store the result of the operation for later use.

The JSON defines the steps in a machine learning model's forward pass and also specifies how the backward pass (which computes gradients for optimization) should be carried out.

By defining the operations and their connections in a JSON file, the graph can be easily constructed and modified, and the computations can be automatically differentiated and parallelized. This representation makes it possible to define a wide variety of models in a modular way, using the building blocks provided by the library.

### Instantiate the architecture

Use a JSON serialization library like Newtonsoft.JSON to deserialize the JSON file to a JsonArchitecure object.

### Instantiate the computational graph

```c#
this.computationGraph = new SelfAttentionMultiLayerLSTMComputationGraph(this);
var zeroMatrixHiddenSize = new Matrix(this.hiddenSize, 1);
this.computationGraph
    .AddIntermediate("inputSequence", x => this.Parameters.InputSequence[x.TimeStep])
    .AddIntermediate("output", x => this.output[x.TimeStep])
    .AddIntermediate("c", x => this.c[x.TimeStep][x.Layer])
    .AddIntermediate("h", x => this.h[x.TimeStep][x.Layer])
    .AddScalar("scaledDotProductScalar", x => 1.0d / Math.Sqrt(this.hiddenSize))
    .AddWeight("Wf", x => this.Wf[x.Layer]).AddGradient("dWf", x => this.dWf[x.Layer])
    .AddWeight("Wi", x => this.Wi[x.Layer]).AddGradient("dWi", x => this.dWi[x.Layer])
    .AddWeight("Wc", x => this.Wc[x.Layer]).AddGradient("dWc", x => this.dWc[x.Layer])
    .AddWeight("Wo", x => this.Wo[x.Layer]).AddGradient("dWo", x => this.dWo[x.Layer])
    .AddWeight("Uf", x => this.Uf[x.Layer]).AddGradient("dUf", x => this.dUf[x.Layer])
    .AddWeight("Ui", x => this.Ui[x.Layer]).AddGradient("dUi", x => this.dUi[x.Layer])
    .AddWeight("Uc", x => this.Uc[x.Layer]).AddGradient("dUc", x => this.dUc[x.Layer])
    .AddWeight("Uo", x => this.Uo[x.Layer]).AddGradient("dUo", x => this.dUo[x.Layer])
    .AddWeight("bf", x => this.bf[x.Layer]).AddGradient("dbf", x => this.dbf[x.Layer])
    .AddWeight("bi", x => this.bi[x.Layer]).AddGradient("dbi", x => this.dbi[x.Layer])
    .AddWeight("bc", x => this.bc[x.Layer]).AddGradient("dbc", x => this.dbc[x.Layer])
    .AddWeight("bo", x => this.bo[x.Layer]).AddGradient("dbo", x => this.dbo[x.Layer])
    .AddWeight("Wq", x => this.Wq[x.Layer]).AddGradient("dWq", x => this.dWq[x.Layer])
    .AddWeight("Wk", x => this.Wk[x.Layer]).AddGradient("dWk", x => this.dWk[x.Layer])
    .AddWeight("Wv", x => this.Wv[x.Layer]).AddGradient("dWv", x => this.dWv[x.Layer])
    .AddWeight("We", x => this.We).AddGradient("dWe", x => this.dWe)
    .AddWeight("be", x => this.be).AddGradient("dbe", x => this.dbe)
    .AddWeight("V", x => this.V).AddGradient("dV", x => this.dV)
    .AddWeight("b", x => this.b).AddGradient("db", x => this.db)
    .AddOperationFinder("i", x => this.computationGraph[$"i_{x.TimeStep}_{x.Layer}"])
    .AddOperationFinder("f", x => this.computationGraph[$"f_{x.TimeStep}_{x.Layer}"])
    .AddOperationFinder("cHat", x => this.computationGraph[$"cHat_{x.TimeStep}_{x.Layer}"])
    .AddOperationFinder("o", x => this.computationGraph[$"o_{x.TimeStep}_{x.Layer}"])
    .AddOperationFinder("embeddedInput", x => this.computationGraph[$"embeddedInput_{x.TimeStep}_0"])
    .AddOperationFinder("hFromCurrentTimeStepAndLastLayer", x => this.computationGraph[$"h_{x.TimeStep}_{this.numLayers - 1}"])
    .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph[$"embeddedInput_{x.TimeStep}_0"] : this.computationGraph[$"h_{x.TimeStep}_{x.Layer - 1}"])
    .AddOperationFinder("previousHiddenState", x => x.TimeStep == 0 ? zeroMatrixHiddenSize : this.computationGraph[$"h_{x.TimeStep - 1}_{x.Layer}"])
    .AddOperationFinder("previousMemoryCellState", x => x.TimeStep == 0 ? zeroMatrixHiddenSize : this.computationGraph[$"c_{x.TimeStep - 1}_{x.Layer}"])
    .ConstructFromArchitecture(jsonArchitecture, this.numTimeSteps, this.numLayers);
```

Operation finders are a key component used to define and locate different operations in a neural network's computational graph. They're essentially functions that link to specific operations at different layers or time steps within the network. This is achieved by mapping string identifiers (IDs) to these operations, which are then used within a JSON architecture to establish the network's structure and sequence of computations. For example, an operation finder could link to a matrix multiplication operation in a specific layer of the network. By using these operation finders, developers can effectively manage complex computational graphs.

### Populate the backward dependency counts

Then populate the backward dependency counts by running the following code. It only has to be run once to set up the backward dependency counts.
```c#
IOperationBase? backwardStartOperation = null;
for (int t = this.Parameters.NumTimeSteps - 1; t >= 0; t--)
{
    backwardStartOperation = this.computationGraph[$"output_t_{t}_0"];
    OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
    await opVisitor.TraverseAsync();
    await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
}
```

### Run the forward pass
```c#
var op = this.computationGraph.StartOperation ?? throw new Exception("Start operation should not be null.");
IOperationBase? currOp = null;
do
{
    var parameters = this.LookupParameters(op);
    var forwardMethod = op.OperationType.GetMethod("Forward") ?? throw new Exception($"Forward method should exist on operation of type {op.OperationType.Name}.");
    forwardMethod.Invoke(op, parameters);
    if (op.ResultToName != null)
    {
        var split = op.ResultToName.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
        var oo = this.computationGraph[MatrixType.Intermediate, split[0], op.LayerInfo];
        op.CopyResult(oo);
    }

    currOp = op;
    if (op.HasNext)
    {
        op = op.Next;
    }
}
while (currOp.Next != null);
```

### Create a loss function
Create a loss function like mean squared error, cross-entropy loss or using policy gradient methods.

Then calculate the gradient of the loss with respect to the output.

Plug the result in as the backward input for the backward start operation.

### Run the backward pass utilizing inherent parallelization
```c#
IOperationBase? backwardStartOperation = null;
for (int t = this.Parameters.NumTimeSteps - 1; t >= 0; t--)
{
    backwardStartOperation = this.computationGraph[$"output_t_{t}_0"];
    if (gradientOfLossWrtOutput[t][0] != 0.0d)
    {
        var backwardInput = new Matrix(1, 1);
        backwardInput[0] = gradientOfLossWrtOutput[t];
        backwardStartOperation.BackwardInput = backwardInput;
        OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
        await opVisitor.TraverseAsync();
        opVisitor.Reset();
        traverseCount++;
    }
}
```

### Using CUDA operations
```c#
Cudablas.Instance.DeviceId = 0; // set the GPU to use, defaults to 0
Cudablas.Instance.Initialize(); // initialize the CUDA library
// ... <Run CUDA operations> ...
Cudablas.Instance.Dispose(); // dispose the CUDA library
```
