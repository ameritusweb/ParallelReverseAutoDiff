# ParallelReverseAutoDiff
Parallel Reverse Mode Automatic Differentiation in C#

![Logo](https://raw.githubusercontent.com/ameritusweb/ParallelReverseAutoDiff/main/docs/parallelreverseautodiff_logo2.png)

[![NuGet version (parallelreverseautodiff)](https://img.shields.io/nuget/v/parallelreverseautodiff?style=flat-square)](https://www.nuget.org/packages/parallelreverseautodiff/)
![Nuget](https://img.shields.io/nuget/dt/parallelreverseautodiff?style=flat-square)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7f9f69794dd74a97aeaac17ebd1580ec)](https://app.codacy.com/gh/ameritusweb/ParallelReverseAutoDiff/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

ParallelReverseAutoDiff is a thread-safe C# library for reverse mode automatic differentiation, optimized for parallel computation. It leverages semaphores and locks to coordinate between threads, ensuring accuracy during gradient accumulation. Each operation in the library is implemented as a node with a forward and a backward function, facilitating efficient calculation of derivatives. A unique aspect of this library is its use of the visitor pattern: it includes a specialized 'Neural Network Visitor' which traverses neural network nodes across different threads. This visitor is responsible for gradient accumulation on nodes shared across multiple threads. This design allows for parallelized computations while maintaining consistency and avoiding race conditions. The result is an efficient, scalable automatic differentiation solution, ideal for machine learning applications and neural network training.

[API Documentation](https://ameritusweb.github.io/ParallelReverseAutoDiff/api/index.html)

## Prerequisites
Download and install the [Cuda Toolkit 12.0](https://developer.nvidia.com/cuda-12-0-0-download-archive) if you want to use the CudaMatrixMultiplyOperation.

## Supported Operations

### Regular Operations
AddGaussianNoiseOperation

AmplifiedSigmoidOperation - Used for gradient amplification.

ApplyDropoutOperation

BatchNormalizationOperation

CudaMatrixMultiplyOperation - Leverages the GPU for fast computation.

GELUOperation

HadamardProductOperation

LayerNormalizationOperation

LeakyReLUOperation

MatrixAddOperation

MatrixAddBroadcastingOperation

MatrixAddThreeOperation

MatrixAverageOperation

MatrixBroadcastOperation

MatrixConcatenateOperation

MatrixMultiplyOperation

MatrixMultiplyScalarOperation

MatrixSumOperation

MatrixTransposeOperation

ReLUOperation

RMSNormOperation

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

## Neural Network Parameters

Each neural network base class has a set of parameters that can be used to configure the neural network. They are as follows:

```csharp
/// <summary>
/// Gets or sets the batch size.
/// </summary>
public int BatchSize { get; set; } = 2;

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

### Build out your neural network model

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

### Instantiate the architecture

Use a JSON serialization library like Newtonsoft.JSON to deserialize the JSON file to a JsonArchitecture object.

There are other JSON architectures available as well.

These include the 'NestedLayersJsonArchitecture', 'DualLayersJsonArchitecture', and 'TripleLayersJsonArchitecture'.

### Instantiate the computational graph

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
    .AddScalar("scaledDotProductScalar", x => 1.0d / Math.Sqrt(this.hiddenSize))
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

### Clip the gradients
```c#
GradientClipper clipper = new GradientClipper(this);
clipper.Clip(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });
```

### Update the weights
```c#
AdamOptimizer optimizer = new AdamOptimizer(this);
optimizer.Optimize(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });
```

### Using CUDA operations
```c#
Cudablas.Instance.DeviceId = 0; // set the GPU to use, defaults to 0
Cudablas.Instance.Initialize(); // initialize the CUDA library
// ... <Run CUDA operations> ...
Cudablas.Instance.Dispose(); // dispose the CUDA library
```

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
