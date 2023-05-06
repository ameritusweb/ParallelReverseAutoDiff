//------------------------------------------------------------------------------
// <copyright file="OperationProcessing.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The operation processing for a feed forward neural network.
    /// </summary>
    public partial class FeedForwardNeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.FeedForwardExample.architecture";

        private string ConvertIdToTimeAndLayer(string id, string[] split, IOperation op)
        {
            if (split.Length == 1)
            {
                if (op.LayerIndex == -1)
                {
                    return id + "_" + op.TimeStepIndex;
                }
                else
                {
                    return id + "_" + op.TimeStepIndex + "_" + op.LayerIndex;
                }
            }

            return id;
        }

        private void SetupDependencies(IOperation op)
        {
            object[] parameters = new object[op.Inputs.Count];
            for (int j = 0; j < op.Inputs.Count; ++j)
            {
                var input = op.Inputs[j];
                var split = input.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                var specificId = this.ConvertIdToTimeAndLayer(input, split, op);
                if (this.operationsMap.ContainsKey(specificId))
                {
                    var inputOp = this.operationsMap[specificId];
                    inputOp.Outputs.Add(op.SpecificId);
                    op.BackwardAdjacentOperations.Add(inputOp);
                    parameters[j] = inputOp;
                    continue;
                }

                string inputName = split[0];

                if (this.inputNameToValueMap.ContainsKey(inputName))
                {
                    int layerIndex = op.LayerIndex;

                    // Get the corresponding value from the dictionary using the input name
                    var p = this.inputNameToValueMap[inputName](layerIndex);
                    if (p is IOperation)
                    {
                        var inputOp = (IOperation)p;
                        inputOp.Outputs.Add(op.SpecificId);
                        op.BackwardAdjacentOperations.Add(inputOp);
                        parameters[j] = inputOp;
                        continue;
                    }
                    else
                    {
                        op.BackwardAdjacentOperations.Add(null);
                    }

                    parameters[j] = p;
                }
                else
                {
                    throw new Exception($"Input name {inputName} not found in value map");
                }
            }

            op.Parameters = parameters;
        }

        private object[] LookupParameters(IOperation op)
        {
            object[] parameters = op.Parameters;
            object[] parametersToReturn = new object[parameters.Length];
            for (int j = 0; j < parameters.Length; ++j)
            {
                if (parameters[j] is IOperation)
                {
                    parametersToReturn[j] = ((IOperation)parameters[j]).GetOutput();
                }
                else
                {
                    parametersToReturn[j] = parameters[j];
                }
            }

            return parametersToReturn;
        }

        private Func<int, object> NameToValueFunc(string name)
        {
            string[] split = name.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
            return this.inputNameToValueMap[split[0]];
        }

        private IOperation ProcessOperation(OperationInfo operation, int layerIndex = -1)
        {
            if (operation == null)
            {
                throw new ArgumentNullException(nameof(operation), $"The parameter {nameof(operation)} cannot be null.");
            }

            string id = operation.Id;
            string typeName = operation.Type;
            string[] inputs = operation.Inputs;

            System.Type operationType = System.Type.GetType($"{NAMESPACE}.{typeName}") ?? throw new Exception($"Unsupported operation type {typeName}");

            var instantiate = operationType.GetMethod("Instantiate");
            if (instantiate == null)
            {
                throw new Exception($"Instantiate method should exist on operation of type {operationType.Name}");
            }

            IOperation op = (IOperation)(instantiate.Invoke(null, new object[] { (NeuralNetwork)this }) ?? throw new Exception("Instantiate method should return a non-null operation."));

            op.OperationType = operationType;
            op.Inputs = inputs.ToList();
            string resultTo = operation.SetResultTo;
            if (resultTo != null)
            {
                op.ResultToName = resultTo;
            }

            string[] gradientResultTo = operation.GradientResultTo;
            if (gradientResultTo != null)
            {
                op.GradientDestinations = new object[gradientResultTo.Length];
                for (int i = 0; i < gradientResultTo.Length; ++i)
                {
                    if (gradientResultTo[i] != null)
                    {
                        op.GradientDestinations[i] = this.NameToValueFunc(gradientResultTo[i])(layerIndex);
                    }
                }
            }

            if (this.priorOperation != null)
            {
                this.priorOperation.Next = op;
            }

            if (this.startOperation == null)
            {
                this.startOperation = op;
            }

            op.Id = id;

            this.priorOperation = op;

            return op;
        }

        private void ProcessAndAddOperation(OperationInfo operationInfo, int layerIndex = -1)
        {
            IOperation op = this.ProcessOperation(operationInfo, layerIndex);
            op.SpecificId = op.Id;

            if (layerIndex != -1)
            {
                op.LayerIndex = layerIndex;
                op.SpecificId += "_" + layerIndex;
            }

            this.operationsMap[op.SpecificId] = op;
        }

        private void CreateOperationsFromJson(string json)
        {
            this.operationsMap = new Dictionary<string, IOperation>();
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.priorOperation = null;
            this.startOperation = null;

            foreach (var timeStep in jsonArchitecture.TimeSteps)
            {
                foreach (var start in timeStep.StartOperations)
                {
                    this.ProcessAndAddOperation(start);
                }

                for (int j = 0; j < this.NumLayers; ++j)
                {
                    foreach (var layer in timeStep.Layers)
                    {
                        foreach (var layerOp in layer.Operations)
                        {
                            this.ProcessAndAddOperation(layerOp, j);
                        }
                    }
                }

                foreach (var end in timeStep.EndOperations)
                {
                    this.ProcessAndAddOperation(end);
                }
            }

            this.priorOperation = null;
        }
    }
}
