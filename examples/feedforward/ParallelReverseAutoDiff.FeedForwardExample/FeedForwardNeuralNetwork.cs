//------------------------------------------------------------------------------
// <copyright file="FeedForwardNeuralNetwork.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.FeedForwardExample.RMAD;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A feed forward neural network.
    /// </summary>
    public partial class FeedForwardNeuralNetwork : NeuralNetwork
    {
        private const string ARCHITECTURE = "FeedForwardArchitecture";
        private EmbeddingLayer embeddingLayer;
        private OutputLayer outputLayer;

        private Dictionary<string, IOperation> operationsMap;
        private IOperation? priorOperation;
        private IOperation? startOperation;
        private Dictionary<string, Func<int, object>> inputNameToValueMap;

        /// <summary>
        /// Initializes a new instance of the <see cref="FeedForwardNeuralNetwork"/> class.
        /// </summary>
        /// <param name="inputSize">The input size.</param>
        /// <param name="hiddenSize">The hidden size.</param>
        /// <param name="outputSize">The output size.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public FeedForwardNeuralNetwork(int inputSize, int hiddenSize, int outputSize, int numLayers, double learningRate, double? clipValue)
        {
            this.InputSize = hiddenSize;
            this.OriginalInputSize = inputSize;
            this.HiddenSize = hiddenSize;
            this.OutputSize = outputSize;
            this.LearningRate = learningRate;
            this.NumLayers = numLayers;
            if (clipValue != null)
            {
                this.ClipValue = clipValue.Value;
            }

            this.HiddenLayers = new HiddenLayer[numLayers];
            this.EmbeddingLayer.Initialize();
            this.EmbeddingLayer.InitializeGradients();
            foreach (var hiddenLayer in this.HiddenLayers)
            {
                hiddenLayer.Initialize();
                hiddenLayer.InitializeGradients();
            }

            this.OutputLayer.Initialize();
            this.OutputLayer.InitializeGradients();

            this.SetupInputNameToValueMap();
        }

        /// <summary>
        /// Gets the input matrix.
        /// </summary>
        public Matrix Input { get; private set; }

        /// <summary>
        /// Gets the embedding layer.
        /// </summary>
        public EmbeddingLayer EmbeddingLayer
        {
            get
            {
                return this.embeddingLayer ??= new EmbeddingLayer(this);
            }
        }

        /// <summary>
        /// Gets or sets the hidden layers.
        /// </summary>
        public HiddenLayer[] HiddenLayers { get; set; }

        /// <summary>
        /// Gets the output layer.
        /// </summary>
        public OutputLayer OutputLayer
        {
            get
            {
                return this.outputLayer ??= new OutputLayer(this);
            }
        }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public Matrix Output { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets the original input size of the neural network.
        /// </summary>
        internal int OriginalInputSize { get; private set; }

        /// <summary>
        /// Gets the input size of the neural network.
        /// </summary>
        internal int InputSize { get; private set; }

        /// <summary>
        /// Gets the hidden size of the neural network.
        /// </summary>
        internal int HiddenSize { get; private set; }

        /// <summary>
        /// Gets the output size of the neural network.
        /// </summary>
        internal int OutputSize { get; private set; }

        /// <summary>
        /// Gets the number of layers of the neural network.
        /// </summary>
        internal int NumLayers { get; private set; }

        /// <summary>
        /// Gets the clip value for the neural network.
        /// </summary>
        internal double ClipValue { get; private set; } = 4d;

        /// <summary>
        /// Gets the Adam iteration for the neural network.
        /// </summary>
        internal double AdamIteration { get; private set; }

        /// <summary>
        /// Initializes the computation graph of the feed forward neural network.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            await this.InitializeComputationGraph();
        }

        /// <summary>
        /// Optimize the neural network.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="target">The target matrix.</param>
        /// <param name="iterationIndex">The iteration index.</param>
        /// <param name="doNotUpdate">Whether or not the parameters should be updated.</param>
        /// <returns>A task.</returns>
        public async Task Optimize(Matrix input, Matrix target, int iterationIndex, bool? doNotUpdate)
        {
            this.Target = target;
            if (doNotUpdate == null)
            {
                doNotUpdate = false;
            }

            this.AdamIteration = iterationIndex + 1;

            await this.AutomaticForwardPropagate(input, doNotUpdate.Value);
        }

        private void ClearState()
        {
            this.EmbeddingLayer.ClearState();
            Parallel.For(0, this.HiddenLayers.Length, (i) =>
            {
                this.HiddenLayers[i].ClearState();
            });
            this.OutputLayer.ClearState();
        }

        private void SetupInputNameToValueMap()
        {
            this.inputNameToValueMap = new Dictionary<string, Func<int, object>>
            {
                { "Input", (_) => this.Input },
                { "Output", (_) => this.Output },
                { "W", (l) => this.HiddenLayers[l].W },
                { "B", (l) => this.HiddenLayers[l].B },
                { "DW", (l) => this.HiddenLayers[l].DW },
                { "DB", (l) => this.HiddenLayers[l].DB },
                { "H", (l) => this.HiddenLayers[l].H },
                { "We", (_) => this.EmbeddingLayer.We },
                { "Be", (_) => this.EmbeddingLayer.Be },
                { "DWe", (_) => this.EmbeddingLayer.DWe },
                { "DBe", (_) => this.EmbeddingLayer.DBe },
                { "V", (_) => this.OutputLayer.V },
                { "Bo", (_) => this.OutputLayer.Bo },
                { "DV", (_) => this.OutputLayer.DV },
                { "DBo", (_) => this.OutputLayer.DBo },
                { "HFromLastLayer", (_) => this.operationsMap["H_" + (this.NumLayers - 1)] },
                { "currentInput", (l) => l == 0 ? this.operationsMap["embeddedInput"] : this.operationsMap["H_" + (l - 1)] },

                // Add other input names and their corresponding getters here
            };
        }

        private async Task InitializeComputationGraph()
        {
            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            this.CreateOperationsFromJson(json);

            var op = this.startOperation;
            if (op == null)
            {
                throw new Exception("Start operation should not be null.");
            }

            IOperation? currOp = null;
            do
            {
                this.SetupDependencies(op);
                this.operationsMap[op.SpecificId] = op;
                currOp = op;
                if (op.HasNext)
                {
                    op = op.Next;
                }
            }
            while (currOp.Next != null);

            IOperation? backwardStartOperation = null;
            backwardStartOperation = this.operationsMap[$"Output"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        private async Task AutomaticForwardPropagate(Matrix input, bool doNotUpdate)
        {
            // Initialize hidden state, gradients, biases, and intermediates
            this.ClearState();

            MatrixUtils.SetInPlace(new[] { this.Input }, new[] { input });
            var op = this.startOperation;
            if (op == null)
            {
                throw new Exception("Start operation should not be null.");
            }

            IOperation? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);
                var forward = op.OperationType.GetMethod("Forward");
                if (forward == null)
                {
                    throw new Exception($"Forward method not found for operation {op.OperationType.Name}");
                }

                forward.Invoke(op, parameters);
                if (op.ResultToName != null)
                {
                    op.ResultTo(this.NameToValueFunc(op.ResultToName));
                }

                this.operationsMap[op.SpecificId] = op;
                currOp = op;
                if (op.HasNext)
                {
                    op = op.Next;
                }
            }
            while (currOp.Next != null);

            await this.AutomaticBackwardPropagate(doNotUpdate);
        }

        private async Task AutomaticBackwardPropagate(bool doNotUpdate)
        {
            var lossFunction = MeanSquaredErrorLossOperation.Instantiate(this);
            var meanSquaredErrorLossOperation = (MeanSquaredErrorLossOperation)lossFunction;
            var loss = meanSquaredErrorLossOperation.Forward(this.Output, this.Target);
            if (loss[0][0] >= 0.0d)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }

            Console.WriteLine($"Mean squared error loss: {loss[0][0]}");
            Console.ForegroundColor = ConsoleColor.White;
            var gradientOfLossWrtOutput = lossFunction.Backward(this.Output).Item1 ?? throw new Exception("Gradient of the loss wrt the output should not be null.");
            int traverseCount = 0;
            IOperation? backwardStartOperation = null;
            backwardStartOperation = this.operationsMap[$"Output"];
            if (gradientOfLossWrtOutput[0][0] != 0.0d)
            {
                backwardStartOperation.BackwardInput = gradientOfLossWrtOutput;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
                traverseCount++;
            }

            if (traverseCount == 0 || doNotUpdate)
            {
                return;
            }

            // Clip gradients and biases to prevent exploding gradients
            this.EmbeddingLayer.ClipGradients();
            Parallel.For(0, this.HiddenLayers.Length, (i) =>
            {
                this.HiddenLayers[i].ClipGradients();
            });

            this.OutputLayer.ClipGradients();

            // Update model parameters using gradient descent
            this.UpdateEmbeddingLayerParametersWithAdam(this.EmbeddingLayer);
            Parallel.For(0, this.HiddenLayers.Length, (i) =>
            {
                this.UpdateHiddenLayerParametersWithAdam(this.HiddenLayers[i]);
            });
            this.UpdateOutputLayerParametersWithAdam(this.OutputLayer);
        }
    }

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
