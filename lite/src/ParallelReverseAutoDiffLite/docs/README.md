# ParallelReverseAutoDiff
Parallel Reverse Mode Automatic Differentiation in C#

![Logo](https://raw.githubusercontent.com/ameritusweb/ParallelReverseAutoDiff/main/docs/parallelreverseautodiff_logo2.png)

[![NuGet version (parallelreverseautodiff)](https://img.shields.io/nuget/v/parallelreverseautodiff?style=flat-square)](https://www.nuget.org/packages/parallelreverseautodiff/)
![Nuget](https://img.shields.io/nuget/dt/parallelreverseautodiff?style=flat-square)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7f9f69794dd74a97aeaac17ebd1580ec)](https://app.codacy.com/gh/ameritusweb/ParallelReverseAutoDiff/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Table of Contents
- [Overview](#overview)
- [ParallelReverseAutoDiffLite](#parallelreverseautodifflite)
- [Building your Neural Network Model](#building-your-neural-network-model)
- [Understanding the JSON Architecture](#understanding-the-json-architecture)
- [Instantiating the Architecture](#instantiating-the-architecture)
- [Instantiating the Computational Graph](#instantiating-the-computational-graph)
- [Populating the Backward Dependency Counts](#populating-the-backward-dependency-counts)
- [Running the Forward Pass](#running-the-forward-pass)
- [Creating a Loss Function](#creating-a-loss-function)
- [Running the Backward Pass](#running-the-backward-pass)
- [Clipping the Gradients](#clipping-the-gradients)
- [Updating the Weights](#updating-the-weights)
- [Using CUDA Operations](#using-cuda-operations)
- [PradOp](#pradop)
- [Customization](#customization)
  - [Custom Neural Network Operations](#custom-neural-network-operations)
- [Examples](#examples)
- [Support Developer](#support-developer)
- [Star the Project](#star-the-project)
- [Reporting Bugs](#reporting-bugs)

ParallelReverseAutoDiff (PRAD) is a thread-safe C# library designed for reverse mode automatic differentiation, optimized for parallel computation and primed for the demands of modern machine learning applications and neural network training. It leverages semaphores and locks to orchestrate between threads, ensuring precision during gradient accumulation.

Upon the realm of code, a gem does shine,

PRAD, a library so divine.

In C# it's crafted, threads align,

Parallel tasks in order, by design.


Reverse mode diff, its core decree,

For neural nets, the key to see.

With batch operations, swift and free,

And ILGPU's might, a sight to be.


From JSON springs architecture grand,

A computational graph, complex and planned.

In the realm of AI, it takes a stand,

A tool of power, in your hand.


So sing we now of PRAD's great glory,

Etched forever in code's vast story.

[API Documentation](https://ameritusweb.github.io/ParallelReverseAutoDiff/api/index.html)

## Overview

Each operation in PRAD is embodied as a node with forward and backward functions, facilitating the efficient calculation of derivatives. This design is particularly beneficial for large-scale problems and complex neural network architectures, where computational efficiency is paramount.

A standout feature of PRAD is its innovative use of the visitor pattern. The library includes a specialized 'Neural Network Visitor' which traverses neural network nodes across different threads. This visitor is tasked with gradient accumulation on nodes shared across multiple threads, allowing for parallelized computations while maintaining consistency and avoiding race conditions.

Moreover, PRAD introduces a data-driven approach to neural network architecture design, allowing for rapid prototyping and experimentation. The library leverages the power of ILGPU, a high-performance GPU-accelerated library for .NET programs, to perform complex computations on the GPU, further enhancing its performance and scalability.

PRAD's dynamic computational graph, constructed from JSON architecture, allows for the efficient computation of gradients, a crucial aspect of the backpropagation process used in training neural networks. This unique blend of features makes PRAD an efficient, scalable, and groundbreaking automatic differentiation solution.

### Prerequisites
Download and install the [Cuda Toolkit 12.0](https://developer.nvidia.com/cuda-12-0-0-download-archive) if you want to use the CudaMatrixMultiplyOperation.

### ParallelReverseAutoDiffLite
ParallelReverseAutoDiffLite (PRADLite) is a lightweight version of PRAD that uses single-precision floating point numbers (float) instead of double-precision floating point numbers (double). This can lead to significant improvements in performance and memory efficiency, especially beneficial for large-scale models and datasets.

#### Key Differences

- **Precision**: PRADLite uses float for all computations, reducing memory footprint and potentially increasing computational speed at the cost of precision.
- **Performance**: Due to the reduced memory requirements and computational overhead, PRADLite can perform faster on both CPUs and GPUs.
- **Compatibility**: PRADLite retains the same API as PRAD, making it easy to switch between the two versions based on your performance needs.

To use PRADLite, simply reference the `ParallelReverseAutoDiffLite` NuGet package in your project instead of `ParallelReverseAutoDiff`.

### Regular Operations
AddGaussianNoiseOperation

AmplifiedSigmoidOperation - Used for gradient amplification.

ApplyDropoutOperation

BatchNormalizationOperation

CosineProjectionOperation

CosineScalingOperation

CudaMatrixMultiplyOperation - Leverages NVIDIA GPUs for fast computation.

DualWeightedOperation

ElementwiseMultiplyAndSumOperation

EmbeddingOperation - Used for word or subword embeddings.

FeatureAggregationOperation - Used for GATs

GELUOperation

GpuMatrixMultiplyAndSumOperation - Leverages NVIDIA, AMD, or Intel GPUs for fast computation.

GpuMatrixMultiplyOperation - Leverages NVIDIA, AMD, or Intel GPUs for fast computation.

GraphAttentionOperation - Used for GATs

HadamardProductOperation

HierarchicalScalingOperation - For increased interpretability.

LayerNormalizationOperation

LeakyReLUOperation

MatrixAddOperation

MatrixAddBroadcastingOperation

MatrixAddThreeOperation

MatrixAverageOperation

MatrixBroadcastOperation

MatrixConcatenateOperation

MatrixDiagonalFilterOperation

MatrixHorizontalConcatenateOperation

MatrixMultiplyOperation

MatrixMultiplyAndSumOperation

MatrixMultiplyAndSumRowsOperation

MatrixMultiplyScalarOperation

MatrixSumOperation

MatrixTransposeOperation

MatrixRowConcatenateOperation

MatrixVectorConcatenateOperation

MatrixVerticalConcatenateOperation

MultiQuerySelfAttentionOperation

PaddingMaskOperation

ReLUOperation

RMSNormOperation

ScaleAndShiftOperation

SigmoidOperation

SineSoftmaxOperation - Reduces the vanishing gradient problem with traditional softmax.

SoftmaxOperation

StretchedSigmoidOperation

SwigLUOperation

SwishOperation

TanhOperation

VariedSoftmaxOperation

### Deep Operations
These types of operations operate on instances of the DeepMatrix class which is a 3-D matrix.
The first dimension is the channel size and the second and third dimensions are the row and column sizes respectively.

DeepBatchNormalizationOperation

DeepConcatenateOperation

DeepConvolutionOperation

DeepLeakyReLUOperation

DeepMatrixElementwiseAddOperation

DeepMatrixElementWiseMultiplySumOperation

DeepMaxPoolOperation

DeepPairwiseAttentionOperation

DeepReLUOperation

DeepScaleAndShiftOperation

FlattenOperation

### Vector Neural Network (VNN) Operations
These types of operations typically operate on instances of the Matrix class where the left half are magnitudes and the right half are angles in radians.
Learn more about Vector Neural Networks [here](https://www.amazon.com/Vector-Neural-Networks-Geometric-Tensors-ebook/dp/B0CXBV3DY5/ref=sr_1_1).

ElementwiseSquareOperation

ElementwiseVectorAddOperation

ElementwiseVectorCartesianSummationOperation

ElementwiseVectorConstituentMultiplyOperation

ElementwiseVectorDecompositionOperation

ElementwiseVectorMiniDecompositionOperation

PairwiseSineSoftmaxOperation

VectorAttentionBinaryOperation

VectorAttentionOperation

VectorizeOperation

### Neural Network Parameters

Each neural network base class has a set of parameters that can be used to configure the neural network. They are as follows:

```csharp
/// <summary>
/// Gets or sets the batch size.
/// </summary>
public int BatchSize { get; set; } = 8;

/// <summary>
/// Gets or sets the dropout rate for the apply dropout operation.
/// </summary>
public double DropoutRate { get; set; } = 0.01d;

/// <summary>
/// Gets or sets the discount factor.
/// </summary>
public double DiscountFactor { get; set; } = 0.99d;

/// <summary>
/// Gets or sets the alpha value for the LeakyReLU operation.
/// </summary>
public double LeakyReLUAlpha { get; set; } = 0.01d;

/// <summary>
/// Gets or sets the learning rate.
/// </summary>
public double LearningRate { get; set; } = 0.001d;

/// <summary>
/// Gets or sets the noise ratio for the AddGaussianNoise operation.
/// </summary>
public double NoiseRatio { get; set; } = 0.01d;

/// <summary>
/// Gets or sets the pool size for the max pool operation.
/// </summary>
public int PoolSize { get; set; } = 2;

/// <summary>
/// Gets or sets the convolution padding for the convolution operation.
/// </summary>
public int ConvolutionPadding { get; set; } = 2;

/// <summary>
/// Gets or sets the beta value for the SwigLU operation.
/// </summary>
public double SwigLUBeta { get; set; } = 1d;

/// <summary>
/// Gets or sets the Adam iteration.
/// </summary>
public double AdamIteration { get; set; } = 1d;

/// <summary>
/// Gets or sets the Adam beta 1.
/// </summary>
public double AdamBeta1 { get; set; } = 0.9d;

/// <summary>
/// Gets or sets the Adam beta 2.
/// </summary>
public double AdamBeta2 { get; set; } = 0.999d;

/// <summary>
/// Gets or sets the Adam epsilon value.
/// </summary>
public double AdamEpsilon { get; set; } = 1E-8d;

/// <summary>
/// Gets or sets the clip value.
/// </summary>
public double ClipValue { get; set; } = 4;

/// <summary>
/// Gets or sets the minimum clip value.
/// </summary>
public double MinimumClipValue { get; set; } = 1E-16;

/// <summary>
/// Gets or sets the number of time steps.
/// </summary>
public int NumTimeSteps { get; set; }

/// <summary>
/// Gets or sets the input sequence.
/// </summary>
public Matrix[] InputSequence { get; set; }

/// <summary>
/// Gets or sets the rewards for policy gradient optimization.
/// </summary>
public List<double> Rewards { get; set; }

/// <summary>
/// Gets or sets the chosen actions for policy gradient optimization.
/// </summary>
public List<Matrix> ChosenActions { get; set; }
```

## Usage

### Building your Neural Network Model

```csharp
var embeddingLayerBuilder = new ModelLayerBuilder(this)
    .AddModelElementGroup("We", new[] { hiddenSize, this.originalInputSize }, InitializationType.Xavier)
    .AddModelElementGroup("be", new[] { hiddenSize, outputSize }, InitializationType.Zeroes);
this.embeddingLayer = embeddingLayerBuilder.Build();

var hiddenLayerBuilder = new ModelLayerBuilder(this)
    .AddModelElementGroup("Wo", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
    .AddModelElementGroup("Uo", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
    .AddModelElementGroup("bo", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
    .AddModelElementGroup("Wi", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
    .AddModelElementGroup("Ui", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
    .AddModelElementGroup("bi", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
    .AddModelElementGroup("Wf", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
    .AddModelElementGroup("Uf", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
    .AddModelElementGroup("bf", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
    .AddModelElementGroup("Wc", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
    .AddModelElementGroup("Uc", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
    .AddModelElementGroup("bc", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
    .AddModelElementGroup("Wq", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
    .AddModelElementGroup("Wk", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
    .AddModelElementGroup("Wv", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier);
this.hiddenLayer = hiddenLayerBuilder.Build();

var outputLayerBuilder = new ModelLayerBuilder(this)
    .AddModelElementGroup("V", new[] { outputSize, hiddenSize }, InitializationType.Xavier)
    .AddModelElementGroup("b", new[] { outputSize, 1 }, InitializationType.Zeroes);
this.outputLayer = outputLayerBuilder.Build();
```

Each model element group needs a unique identifier, a size array, and an initialization type.

The three possible initialization types are Xavier, He, and Zeroes.

The group consists of weights, gradients, and moments for Adam optimization.

The initialization type is used to initialize the model element group's weights.

The model element group's elements are stored in a matrix whose size is specified by the size array.

In this example, for the hidden layer, the first dimension is the number of layers and the second and third dimensions are the row and column sizes respectively.

### Understanding the JSON Architecture

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

*   "timeSteps": This is an array that represents the sequence of computational operations. Each element in the array is an object that corresponds to a computational timestep.

*   "startOperations": This is an array that defines the initial operations for the current timestep.

*   "layers": This represents a sequence of operations corresponding to the layers of the network. Each operation in a layer is a step in the computation, and the order of operations matters, as some operations depend on the results of previous operations. 

*   "endOperations": This is an array that defines the final operations for the current timestep.

Each operation object in "startOperations", "layers", or "endOperations" has several fields:

*   "id": This is a unique identifier for the operation.

*   "description": This is a human-readable description of what the operation does.

*   "type": This specifies the type of the operation.

*   "inputs": This is an array that lists the inputs for the operation. These are the identifiers of other nodes in the computational graph. The identifiers are either defined in the computational graph, or in operation finders declared in code when building an instance of the ComputationGraph class, for example the SelfAttentionMultiLayerLSTMComputationGraph class which is a subclass of ComputationGraph.

*   "gradientResultTo": This is an array that specifies where the results of the backward pass (i.e., the computed gradients) should be stored. A null value means that the gradient with respect to the input is not stored. There is an implicit mapping between the gradient and the input based on its position in the array.

*   "setResultTo": This is used to store the result of the operation for later use.

The JSON defines the steps in a machine learning model's forward pass and also specifies how the backward pass (which computes gradients for optimization) should be carried out.

By defining the operations and their connections in a JSON file, the graph can be easily constructed and modified, and the computations can be automatically differentiated and parallelized. This representation makes it possible to define a wide variety of models in a modular way, using the building blocks provided by the library.

### Instantiating the Architecture

Use a JSON serialization library like Newtonsoft.JSON to deserialize the JSON file to a JsonArchitecture object.

There are other JSON architectures available as well.

These include the 'NestedLayersJsonArchitecture', 'DualLayersJsonArchitecture', and 'TripleLayersJsonArchitecture'.

### Instantiating the Computational Graph

```c#
// Retrieve the matrices from the model layers created by the model layer builder.
var we = this.embeddingLayer.WeightMatrix("We");
var be = this.embeddingLayer.WeightMatrix("be");

var dwe = this.embeddingLayer.GradientMatrix("We");
var dbe = this.embeddingLayer.GradientMatrix("be");

var wf = this.hiddenLayer.WeightDeepMatrix("Wf");
var wi = this.hiddenLayer.WeightDeepMatrix("Wi");
var wc = this.hiddenLayer.WeightDeepMatrix("Wc");
var wo = this.hiddenLayer.WeightDeepMatrix("Wo");

var dwf = this.hiddenLayer.GradientDeepMatrix("Wf");
var dwi = this.hiddenLayer.GradientDeepMatrix("Wi");
var dwc = this.hiddenLayer.GradientDeepMatrix("Wc");
var dwo = this.hiddenLayer.GradientDeepMatrix("Wo");

var uf = this.hiddenLayer.WeightDeepMatrix("Uf");
var ui = this.hiddenLayer.WeightDeepMatrix("Ui");
var uc = this.hiddenLayer.WeightDeepMatrix("Uc");
var uo = this.hiddenLayer.WeightDeepMatrix("Uo");

var duf = this.hiddenLayer.GradientDeepMatrix("Uf");
var dui = this.hiddenLayer.GradientDeepMatrix("Ui");
var duc = this.hiddenLayer.GradientDeepMatrix("Uc");
var duo = this.hiddenLayer.GradientDeepMatrix("Uo");

var bf = this.hiddenLayer.WeightDeepMatrix("bf");
var bi = this.hiddenLayer.WeightDeepMatrix("bi");
var bc = this.hiddenLayer.WeightDeepMatrix("bc");
var bo = this.hiddenLayer.WeightDeepMatrix("bo");

var dbf = this.hiddenLayer.GradientDeepMatrix("bf");
var dbi = this.hiddenLayer.GradientDeepMatrix("bi");
var dbc = this.hiddenLayer.GradientDeepMatrix("bc");
var dbo = this.hiddenLayer.GradientDeepMatrix("bo");

var wq = this.hiddenLayer.WeightDeepMatrix("Wq");
var wk = this.hiddenLayer.WeightDeepMatrix("Wk");
var wv = this.hiddenLayer.WeightDeepMatrix("Wv");

var dwq = this.hiddenLayer.GradientDeepMatrix("Wf");
var dwk = this.hiddenLayer.GradientDeepMatrix("Wi");
var dwv = this.hiddenLayer.GradientDeepMatrix("Wc");

var v = this.outputLayer.WeightMatrix("V");
var b = this.outputLayer.WeightMatrix("b");

var dv = this.outputLayer.GradientMatrix("V");
var db = this.outputLayer.GradientMatrix("b");

// Instantiate the computation graph
this.computationGraph = new SelfAttentionMultiLayerLSTMComputationGraph(this);
var zeroMatrixHiddenSize = new Matrix(this.hiddenSize, 1);
this.computationGraph
    .AddIntermediate("inputSequence", x => this.Parameters.InputSequence[x.TimeStep])
    .AddIntermediate("output", x => this.output[x.TimeStep])
    .AddIntermediate("c", x => this.c[x.TimeStep][x.Layer])
    .AddIntermediate("h", x => this.h[x.TimeStep][x.Layer])
    .AddScalar("scaledDotProductScalar", x => 1.0d / PradMath.Sqrt(this.hiddenSize))
    .AddWeight("Wf", x => wf[x.Layer]).AddGradient("dWf", x => dwf[x.Layer])
    .AddWeight("Wi", x => wi[x.Layer]).AddGradient("dWi", x => dwi[x.Layer])
    .AddWeight("Wc", x => wc[x.Layer]).AddGradient("dWc", x => dwc[x.Layer])
    .AddWeight("Wo", x => wo[x.Layer]).AddGradient("dWo", x => dwo[x.Layer])
    .AddWeight("Uf", x => uf[x.Layer]).AddGradient("dUf", x => duf[x.Layer])
    .AddWeight("Ui", x => ui[x.Layer]).AddGradient("dUi", x => dui[x.Layer])
    .AddWeight("Uc", x => uc[x.Layer]).AddGradient("dUc", x => duc[x.Layer])
    .AddWeight("Uo", x => uo[x.Layer]).AddGradient("dUo", x => duo[x.Layer])
    .AddWeight("bf", x => bf[x.Layer]).AddGradient("dbf", x => dbf[x.Layer])
    .AddWeight("bi", x => bi[x.Layer]).AddGradient("dbi", x => dbi[x.Layer])
    .AddWeight("bc", x => bc[x.Layer]).AddGradient("dbc", x => dbc[x.Layer])
    .AddWeight("bo", x => bo[x.Layer]).AddGradient("dbo", x => dbo[x.Layer])
    .AddWeight("Wq", x => wq[x.Layer]).AddGradient("dWq", x => dwq[x.Layer])
    .AddWeight("Wk", x => wk[x.Layer]).AddGradient("dWk", x => dwk[x.Layer])
    .AddWeight("Wv", x => wv[x.Layer]).AddGradient("dWv", x => dwv[x.Layer])
    .AddWeight("We", x => we).AddGradient("dWe", x => dwe)
    .AddWeight("be", x => be).AddGradient("dbe", x => dbe)
    .AddWeight("V", x => v).AddGradient("dV", x => dv)
    .AddWeight("b", x => b).AddGradient("db", x => db)
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

### Populating the Backward Dependency Counts

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

### Running the Forward Pass
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

### Creating a Loss Function
Create a loss function like mean squared error, cross-entropy loss or using policy gradient methods.

Then calculate the gradient of the loss with respect to the output.

Plug the result in as the backward input for the backward start operation.

### Running the Backward Pass
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

### Clipping the Gradients
```c#
GradientClipper clipper = new GradientClipper(this);
clipper.Clip(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });
```

### Updating the Weights
```c#
AdamOptimizer optimizer = new AdamOptimizer(this);
optimizer.Optimize(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });
```

### Using CUDA Operations
```c#
Cudablas.Instance.DeviceId = 0; // set the GPU to use, defaults to 0
Cudablas.Instance.Initialize(); // initialize the CUDA library
// ... <Run CUDA operations> ...
Cudablas.Instance.Dispose(); // dispose the CUDA library
```

## PradOp

`PradOp` is a 'pretty rad' automatic differentiation operation class in the ParallelReverseAutoDiff (PRAD) library, providing a lightweight reverse-mode automatic differentiation implementation. It supports various tensor operations and manages the computation graph for efficient gradient calculations. It is meant to be used within both the Forward and Backward functions for custom neural network operations to simplify the code, especially the backward pass, by leveraging automatic differentiation.

### Features

- **Efficient Tensor Operations**: Leverages the Tensor class for fast, vectorized computations using the math kernel library (MKL.NET).
- **Automatic Differentiation**: Records operations for both forward and backward passes.
- **Parallelization**: Supports parallel execution of multiple operations for improved performance.
- **Flexible Architecture**: Can be used as a standalone mini-computation graph or integrated into larger systems.
- **Rich Operation Set**: Includes element-wise operations, reshaping, transposing, gathering, and more.
- **Branching Support**: Allows for creation of complex computational graphs with branching paths.
- **Gradient Accumulation**: Designed to work with systems that use gradient accumulation and optimization.
- **Custom Operations**: Supports implementation of custom operations with automatic gradient computation.

### Key Components

#### PradOp

- Manages the computational graph and operations.
- Handles forward and backward passes.
- Supports branching and splitting for complex graph structures.

#### PradResult

- Encapsulates the result of a computation.
- Provides methods for backpropagation and chaining operations.

### Constructor 

```csharp 
public PradOp(Tensor seed) 
``` 

Creates a new instance of the `PradOp` class with a seed tensor. 

- `seed`: The initial tensor to start computations with. 

### Properties 

| Property | Type | Description | 
|----------|------|-------------| 
| `UpstreamGradient` | `Tensor` | Gets or sets the upstream gradient for backpropagation. | 
| `SeedGradient` | `Tensor` | Gets or sets the gradient of the seed tensor. | 
| `IsDependentBranch` | `bool` | Indicates whether this is a dependent branch in the computation graph. | 
| `CurrentShape` | `int[]` | Gets the shape of the current tensor. | 
| `Id` | `Guid` | Gets the unique identifier of the PradOp instance. |  
| `Result` | `Tensor?` | Gets the result of the computation. | 

### Methods 

#### Tensor Operations 

| Method | Description | 
|--------|-------------| 
| `Add(Tensor tensor)` | Adds the given tensor to the current tensor element-wise. | 
| `Sub(Tensor tensor)` | Subtracts the given tensor from the current tensor element-wise. | 
| `SubFrom(Tensor tensor)` | Subtracts the current tensor from the given tensor element-wise. | 
| `Mul(Tensor tensor)` | Multiplies the current tensor by the given tensor element-wise. | 
| `Div(Tensor tensor)` | Divides the current tensor by the given tensor element-wise. | 
| `DivInto(Tensor tensor)` | Divides the given tensor by the current tensor element-wise. | 
| `MatMul(Tensor tensor)` | Performs matrix multiplication of the current tensor with the given tensor. | 
| `Sin()` | Computes the sine of each element in the current tensor. | 
| `Cos()` | Computes the cosine of each element in the current tensor. | 
| `Atan2(Tensor tensor)` | Computes the arctangent of the quotient of the current tensor and the given tensor. | 
| `Square()` | Computes the square of each element in the current tensor. | 
| `SquareRoot()` | Computes the square root of each element in the current tensor. | 
| `SumRows()` | Sums the rows of the current tensor. | 
| `Exp()` | Computes the exponential of each element in the current tensor. | 
| `Ln()` | Computes the natural logarithm of each element in the current tensor. | 
| `Log()` | Computes the base-10 logarithm of each element in the current tensor. | 
| `Mean(int axis)` | Computes the mean along the specified axis in the current tensor. | 
| `Reciprocal()` | Computes the reciprocal of each element in the current tensor. | 
| `Clip(double min, double max)` | Clips values to the specified range. | 
| `Exclude(double min, double max)` | Excludes values within the specified range. | 
| `Sum(int[] axes)` | Sums the tensor along specified axes. | 
| `BroadcastTo(int[] newShape)` | Broadcasts the tensor to a new shape. |
| `LessThan(Tensor tensor)` | Performs element-wise "less than" comparison. |
| `Where(Tensor condition, Tensor other)` | Selects elements based on a condition tensor. |
| `Modulus(Tensor tensor)` | Performs element-wise modulus operation. |
| `ExtractPatches(int[] filterSize, int[] strides, string padding)` | Extracts patches from a tensor for im2col. |
| `Pow(Tensor tensor)` | Performs an element-wise power operation on the current tensor with the provided exponent. |

#### Tensor Manipulation 

| Method | Description | 
|--------|-------------| 
| `Indexer(params string[] indices)` | Slices the tensor using the given indices. | 
| `Reshape(int[] newShape)` | Reshapes the current tensor to the specified shape. | 
| `Transpose(params int[] permutations)` | Transposes the current tensor according to the given permutations. | 
| `Split(int groupSize, int axis = 0)` | Splits the tensor into multiple tensors along the specified axis. | 
| `Tile(int[] multiples)` | Tiles the tensor along each dimension according to the given multiples. | 
| `Gather(Tensor indices, int axis = 0)` | Gathers slices from the tensor along the specified axis. | 
| `GatherNd(Tensor indices)` | Gathers slices from the tensor using multidimensional indices. | 
| `Slice(int[] begin, int[] size, int[]? strides = null)` | Extracts a slice from the tensor. | 
| `Stack(Tensor[] tensors, int axis = 0)` | Stacks the current tensor with other tensors along a new axis. | 
| `Concat(Tensor[] tensors, int axis = 0)` | Concatenates the current tensor with other tensors along a specified axis. | 
| `ExpandDims(int axis = -1)` | Expands the dimensions of the tensor along the specified axis. | 

#### Computation Graph Management 

| Method | Description | 
|--------|-------------| 
| `Branch()` | Creates a new branch in the computation graph. | 
| `DoParallel(params Func<PradOp, PradResult>[] operations)` | Executes multiple operations in parallel. | 
| `Back(Tensor tensor)` | Initiates backpropagation with the given upstream gradient. | 
| `Back()` | Computes backpropagation to accumulate gradients. | 
| `BranchStack(int n)` | Creates a specified number (n) of branches from the current PradOp instance and returns a BranchStack object for managing these branches. This allows you to easily pop branches as needed. |

### Static Operations 

`PradOp` provides several static operation delegates that can be used with the `Then` method of `PradResult`: 

- `SquareRootOp` 
- `AddOp` 
- `MulOp` 
- `SubOp`
- `SubFromOp`
- `DivOp`
- `DivIntoOp`
- `MatMulOp`
- `ExpandDimsOp`
- `SinOp` 
- `CosOp`
- `ReciprocalOp`
- `ExpOp`
- `LnOp`
- `LogOp`
- `MeanOp`
- `GatherOp`
- `GatherNdOp` 
- `SumRowsOp` 
- `SquareOp` 
- `Atan2Op` 
- `StackOp` 
- `ConcatOp`
- `IndexerOp`
- `ReshapeOp`
- `TransposeOp`
- `TileOp`
- `ClipOp`
- `ExcludeOp`
- `SumOp`
- `BroadcastToOp`
- `LessThanOp`
- `WhereOp`
- `ModulusOp`
- `ExtractPatchesOp`
- `PowOp`

### PradResult.Then Method

The `Then` method in `PradResult` is a powerful feature that allows for elegant chaining of operations in the computational graph. It provides a fluent interface for applying successive operations to the result of previous computations.

#### Method Signatures

```csharp
public PradResult Then(Delegate operation, Tensor? other = null)
public PradResult Then(Delegate operation, Tensor[] others, int axis = 0)
public PradResult Then(Func<PradResult[], PradResult> operation)
public PradResult[] Then(Func<PradResult[], PradResult[]> operation)
```

#### Functionality

1. **Chaining Operations**: The `Then` method allows you to apply a new operation to the result of the previous operation. This creates a chain of operations that can be read from left to right, improving code readability.

2. **Static Operation Delegates**: The method uses static operation delegates defined in `PradOp` (like `AddOp`, `MulOp`, `SinOp`, etc.) to determine which operation to apply. These static delegates act as keys to retrieve the corresponding instance method.

3. **Flexible Input**: The method can handle operations that require no additional input, a single additional tensor, or multiple additional tensors and an axis.

4. **Dynamic Dispatch**: The method uses the `GetOperation<T>` method of `PradOp` to dynamically retrieve the correct instance method based on the static delegate provided.

 ### PradResult.ThenParallel Method

 The `ThenParallel` method allows for parallel execution of multiple operations on the same PradResult. This is useful for creating branching computational graphs where multiple operations are performed on the same input.

 #### Method Signature

 ```csharp
 public PradResult[] ThenParallel(params Func<PradResult, PradResult>[] operations)
 ```

 #### Functionality

 1. **Parallel Execution**: The method executes multiple operations in parallel, each operating on a copy of the current PradResult.
 2. **Branching**: It creates multiple branches in the computation graph, allowing for different operations to be applied to the same input.
 3. **Result Aggregation**: Returns an array of PradResult instances, one for each parallel operation.

### Usage Examples 

Here are some examples of how to use `PradOp`: 

```csharp 
// Create a PradOp instance with a seed tensor 
var seed = new Tensor(new double[,] { { 1, 2 }, { 3, 4 } }); 
var op = new PradOp(seed); 

// Perform operations 
var result = op.Add(new Tensor(new double[,] { { 5, 6 }, { 7, 8 } })) 
               .Then(PradOp.SquareRootOp) 
               .Then(PradOp.MulOp, new Tensor(new double[,] { { 2, 2 }, { 2, 2 } })); 

// Compute gradients 
op.Back(new Tensor(new double[,] { { 1, 1 }, { 1, 1 } })); 

// Access gradients
var seedGradient = op.SeedGradient;
```

#### Chaining Operations

PradResult allows for elegant chaining of operations:

```csharp
var x = new PradOp(inputTensor);
var y = someOtherTensor;
var result = x.SquareRoot().Then(PradOp.Add, y);
```

#### Parallel Operations

PradOp supports parallel execution of multiple operations:

```csharp -->
var (result1, result2) = pradOp.DoParallel( 
    x => x.Sin(), 
    x => x.Cos()
);
```

Here is a neural network layer with multiple activations:

This example demonstrates how to use ThenParallel and the Then overloads to compute a neural network layer with multiple activation functions in parallel.

```csharp
// Define input and weights
var input = new Tensor(new int[] { 1, 4 }, new double[] { 0.1, 0.2, 0.3, 0.4 });
var weights = new Tensor(new int[] { 4, 3 }, new double[] { 
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6,
    0.7, 0.8, 0.9,
    1.0, 1.1, 1.2
});
var bias = new Tensor(new int[] { 1, 3 }, new double[] { 0.1, 0.2, 0.3 });

// Create PradOp instance
var pradOp = new PradOp(input);

PradResult? weightsResult = null;
PradResult? biasResult = null;

// Compute layer output with multiple activations
var result = pradOp.MatMul(weights)
    .Then(result => {
        weightsResult = result;
        return result.Then(PradOp.AddOp, bias);
    })
    .Then(result => {
        biasResult = result;
        return result.ThenParallel(
            result => result.Then(PradOp.SinOp),       // Sine activation
            result => result.Then(PradOp.ReciprocalOp).Then(PradOp.AddOp, new Tensor(new int[] { 1, 3 }, 1)),
            result => result.Then(PradOp.ExpOp));        // Exponential activation
    })
    .Then(activations => {
        // Compute weighted sum of activations
        var weights = new Tensor(new int[] { 3 }, new double[] { 0.3, 0.3, 0.4 });
        return activations
            .Select((act, i) => act.PradOp.Mul(weights.Indexer($"{i}").BroadcastTo(new int[] { 1, 3 })))
            .Aggregate((a, b) => a.PradOp.Add(b.Result));
    });

// Compute gradient
var upstreamGradient = new Tensor(new int[] { 1, 3 }, new double[] { 1, 1, 1 });
var gradient = pradOp.Back(upstreamGradient);

// Access results and gradients
Console.WriteLine("Layer output: " + result.Result);
Console.WriteLine("Input gradient: " + gradient);
```

This example showcases:

1. **Matrix Multiplication and Bias Addition**: Simulating a basic neural network layer computation.
2. **Parallel Activation Functions**: Using ThenParallel to apply multiple activation functions to the layer output simultaneously.
3. **Result Aggregation**: Using the Then method to combine the results of multiple activations with a weighted sum.
4. **Gradient Computation**: Demonstrating how gradients can be computed through this complex computation graph.

This example illustrates how ThenParallel and the Then overloads can be used to create more complex and flexible computational graphs, such as those found in advanced neural network architectures with multiple parallel pathways.

#### Combining LessThan, Where, and Modulus

```csharp
// Create input tensors
var x = new Tensor(new int[] { 2, 3 }, new double[] { 1, 2, 3, 4, 5, 6 });
var y = new Tensor(new int[] { 2, 3 }, new double[] { 3, 3, 3, 3, 3, 3 });
var pradOp = new PradOp(x);

// Perform operations
var result = pradOp
    .LessThan(y)  // Check which elements of x are less than 3
    .Then(lessThanResult => {
        var lessThanResultBranch = lessThanResult.PradOp.Branch();
        var modulusResult = lessThanResult.PradOp.Modulus(new Tensor(new int[] { 2, 3 }, new double[] { 2, 2, 2, 2, 2, 2 }));
        return modulusResult.PradOp.Where(lessThanResultBranch.BranchInitialTensor, y);
    });

// Compute gradients
var upstreamGradient = new Tensor(new int[] { 2, 3 }, new double[] { 1, 1, 1, 1, 1, 1 });
var gradient = pradOp.Back(upstreamGradient);
```

#### Custom Operations 

PradOp allows you to define custom operations with their own forward and backward passes. Here's an example of a custom sigmoid operation: 

```csharp 
public PradResult CustomSigmoid() 
{ 
    return this.CustomOperation( 
        operation: input =>  
        { 
            var result = new Tensor(input.Shape); 
            for (int i = 0; i < input.Data.Length; i++) 
            { 
                result.Data[i] = 1 / (1 + PradMath.Exp(-input.Data[i])); 
            } 
            return result; 
        }, 
        reverseOperation: (input, output, upstreamGrad) =>  
        { 
            var gradient = new Tensor(input.Shape); 
            for (int i = 0; i < input.Data.Length; i++) 
            { 
                gradient.Data[i] = output.Data[i] * (1 - output.Data[i]) * upstreamGrad.Data[i]; 
            } 
            return new[] { gradient }; 
        }, 
        outputShape: this.currentTensor.Shape 
    ); 
} 
``` 

Usage: 

```csharp 
var pradOp = new PradOp(inputTensor); 
var result = pradOp.CustomSigmoid(); 
var gradient = pradOp.Back(upstreamGradient); 
``` 

#### Branching 

PradOp supports creating complex computational graphs with branching paths. Here's an example: 

```csharp 
var pradOp = new PradOp(inputTensor); 
var branch = pradOp.Branch(); 

var result1 = pradOp.Sin(); 
var result2 = branch.Cos(); 

var combinedResult = pradOp.Add(result2.Result); 
var gradient = pradOp.Back(upstreamGradient); 
``` 

#### BranchStack

The `BranchStack(int n)` method is designed to streamline the management of multiple branches within the computation graph. It creates `n` branches from the current `PradOp` instance and encapsulates them in a `BranchStack` object. This object provides a `Pop()` method to retrieve and work with individual branches in a controlled and orderly manner.

```csharp
var tBranches = t.BranchStack(4);

var t2 = t.Square();
var t3 = t2.Then(PradOp.MulOp, tBranches.Pop().BranchInitialTensor);
var t4 = t3.Then(PradOp.MulOp, tBranches.Pop().BranchInitialTensor);

var mt = tBranches.Pop().SubFrom(new Tensor(t.CurrentShape, 1.0));
var mt2 = mt.Square();
var mt3 = mt2.Then(PradOp.MulOp, tBranches.Pop().BranchInitialTensor);
```

#### Splitting 

PradOp allows you to split tensors and perform operations on the split parts. Here's an example: 

```csharp 
var pradOp = new PradOp(inputTensor); // Assume inputTensor has shape [4, 10] 
var (leftHalf, rightHalf) = pradOp.Split(5, axis: 1); // Split along the second dimension 

var processedLeft = leftHalf.Square(); 
var processedRight = rightHalf.SquareRoot(); 

var recombined = leftHalf.Stack(new[] { processedRight.Result }, axis: 1); 
var gradient = recombined.Back(upstreamGradient); 
``` 

These examples demonstrate the flexibility of PradOp in handling complex computational graphs, including custom operations, branching, and splitting/recombining tensors. The automatic differentiation system takes care of computing the correct gradients through these complex structures.

## Customization
The ParallelReverseAutoDiff (PRAD) library is designed with customization at its core.

Understanding that the world of machine learning and neural networks is continually evolving, the library allows users to define their own neural network operations.

This feature provides an immense level of flexibility and control over the architecture and behavior of the networks, making it adaptable to both traditional and experimental models.

### Custom Neural Network Operations
One of the standout features of PRAD is the ability to create custom operations. 

These operations can encapsulate any computation or processing steps, including but not limited to, complex forward and backward calculations, and operations involving matrices, vectors, or scalars.

Creating a custom operation requires extending the Operation abstract class, which involves implementing two key methods:

*   Forward(): This method is used to describe how your operation behaves during the forward pass of the neural network. It takes as input the relevant data, processes it as per the custom-defined operation, and produces the output.

*   Backward(): This method is responsible for defining how your operation behaves during the backward pass of the neural network, i.e., how it contributes to the gradients during backpropagation. It receives the gradient of the output and uses it to compute the gradients of its inputs.

Let's look at an example custom operation, MatrixAverageOperation, which calculates the average of feature vectors across a matrix:

```c#
public class MatrixAverageOperation : Operation
{
    private Matrix input;

    public static IOperation Instantiate(NeuralNetwork net)
    {
        return new MatrixAverageOperation();
    }

    public Matrix Forward(Matrix input)
    {
        int numRows = input.Rows;
        this.input = input;
        this.Output = new Matrix(numRows, 1);

        for (int i = 0; i < numRows; i++)
        {
            this.Output[i][0] = input[i].Average();
        }

        return this.Output;
    }

    public override BackwardResult Backward(Matrix dOutput)
    {
        int numRows = dOutput.Length;
        int numCols = this.input.Cols;

        Matrix dInput = new Matrix(numRows, numCols);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                dInput[i][j] = dOutput[i][0] / numCols;
            }
        }

        return new BackwardResultBuilder()
            .AddInputGradient(dInput)
            .Build();
    }
}
```
In this example, the Forward method calculates the average of the features for each path, while the Backward method spreads the gradient evenly across the features.

This level of customization allows PRAD to be a versatile tool in the field of machine learning, capable of being tailored to a wide range of tasks, datasets, and innovative architectures.

To register your custom operation with the computation graph, add this to your computation graph class:

```c#
protected override Type TypeRetrieved(string type)
{
    var retrievedType = base.TypeRetrieved(type);
    var customType = Type.GetType("ParallelReverseAutoDiff.RMAD." + type); // replace with your own namespace
    if (customType == null)
    {
        return retrievedType;
    } else
    {
        return customType;
    }
}
```

## Examples

To help you get started with ParallelReverseAutoDiff, we've provided a set of examples in the repository. These examples demonstrate various use cases and features of the library.

You can find these examples in the [examples folder](https://github.com/ameritusweb/ParallelReverseAutoDiff/tree/main/examples) of the repository.

## Support Developer
[!["Buy Me A Coffee"](https://raw.githubusercontent.com/ameritusweb/ParallelReverseAutoDiff/main/docs/orange_img.png)](https://www.buymeacoffee.com/ameritusweb)

## Star the Project

Give it a :star: Star!

## Reporting Bugs

Drop to [Issues](https://github.com/ameritusweb/ParallelReverseAutoDiff/issues)

Or: ameritusweb@gmail.com

Thanks in advance!
