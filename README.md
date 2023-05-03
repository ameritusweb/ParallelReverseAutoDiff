# ParallelReverseAutoDiff
Parallel Reverse Mode Automatic Differentiation in C#

[![NuGet version (parallelreverseautodiff)](https://img.shields.io/nuget/v/parallelreverseautodiff?style=flat-square)](https://www.nuget.org/packages/parallelreverseautodiff/)

ParallelAutoDiff is a thread-safe C# library for reverse mode automatic differentiation, optimized for parallel computation. It leverages semaphores and locks to coordinate between threads, ensuring accuracy during gradient accumulation. Each operation in the library is implemented as a node with a forward and a backward function, facilitating efficient calculation of derivatives. A unique aspect of this library is its use of the visitor pattern: it includes a specialized 'Neural Network Visitor' which traverses neural network nodes across different threads. This visitor is responsible for gradient accumulation on nodes shared across multiple threads. This design allows for parallelized computations while maintaining consistency and avoiding race conditions. The result is an efficient, scalable automatic differentiation solution, ideal for machine learning applications and neural network training.

## Supported Operations
AmplifiedSigmoidOperation - Used for gradient amplification

ApplyDropoutOperation

HadamardProductOperation

LayerNormalizationOperation

LeakyReLUOperation

MatrixAddOperation

MatrixAddThreeOperation

MatrixMultiplyOperation

MatrixMultiplyScalarOperation

MatrixTransposeOperation

ReLUOperation

SigmoidOperation

SoftmaxOperation

StretchedSigmoidOperation

TanhOperation

## Usage

### Create architecture JSON file

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

### Instantiate the architecture

Use a JSON serialization library like Newtonsoft.JSON to deserialize the JSON file to a JSONArchitecure object.

### Instantiate and populate the operations

Instantiate each operation based on its type. Then set the Next property of each operation to be the next operation in the forward pass.

Add each operation that is backwards in the computation graph to the BackwardAdjacentOperations property of an operation. BackwardAdjacentOperations is just a list of operations.

Add instances of the gradients to send the result to, to the GradientDestinations array. If there is no gradient result for a certain input, add null.

Then populate the backward dependency counts by running the following code. It only has to be run once to set up the backward dependency counts.
```c#
for (int t = numTimeSteps - 1; t >= 0; t--) // if there are multiple timesteps
{
    backwardStartOperation = operationsMap[$"output_t_{t}"]; // the backward start operation
    OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
    await opVisitor.TraverseAsync(); // sets the backward dependency counts
    await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
}
```

### Run the forward pass
```c#
var op = startOperation; // the start operation
IOperation currOp = null;
do
{
    var parameters = LookupParameters(op); // lookup the parameters
    op.OperationType.GetMethod("Forward").Invoke(op, parameters); // call the forward function
    if (op.ResultToName != null)
    {
        op.ResultTo(NameToValueFunc(op.ResultToName)); // send the result to the appropriate object
    }
    operationsMap[op.SpecificId] = op;
    currOp = op;
    if (op.HasNext)
        op = op.Next;
} while (currOp.Next != null);
```

### Run the backward pass utilizing inherent parallelization
```c#
for (int t = numTimeSteps - 1; t >= 0; t--)
{
    backwardStartOperation = operationsMap[$"output_t_{t}"];
    if (gradientOfLossWrtOutput[t][0] != 0.0d)
    {
        backwardStartOperation.BackwardInput = new double[][] { gradientOfLossWrtOutput[t] };
        OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
        await opVisitor.TraverseAsync();
        opVisitor.Reset();
        traverseCount++;
    }
}
```
