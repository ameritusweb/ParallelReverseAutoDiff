//------------------------------------------------------------------------------
// <copyright file="FeedForwardNeuralNetwork.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.FeedForwardExample.RMAD;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A feed forward neural network.
    /// </summary>
    public partial class FeedForwardNeuralNetwork : NeuralNetwork
    {
        private const string ARCHITECTURE = "FeedForwardArchitecture";
        private readonly double clipValue = 4.0d;
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
                this.clipValue = clipValue.Value;
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
        /// Initializes the computation graph of the feed forward neural network.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            await this.InitializeComputationGraph();
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
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        private async Task AutomaticForwardPropagate(Matrix input, Matrix target, bool doNotUpdate)
        {
            // Initialize memory cell, hidden state, gradients, biases, and intermediates
            this.ClearState();

            MatrixUtils.SetInPlace(this.inputSequence, inputSequence);
            this.Target = target;
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

        private async Task AutomaticBackwardPropagate(bool doNotUpdate = false)
        {
            var lossFunction = MeanSquaredErrorLossOperation.Instantiate(this);
            var policyGradientLossOperation = (MeanSquaredErrorLossOperation)lossFunction;
            var loss = policyGradientLossOperation.Forward(this.Output);
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
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
                traverseCount++;
            }

            if (traverseCount == 0 || doNotUpdate)
            {
                return;
            }

            // Clip gradients and biases to prevent exploding gradients

            // Update model parameters using gradient descent
            this.UpdateParametersWithAdam(this.dWi, this.dWf, this.dWo, this.dWc, this.dUi, this.dUf, this.dUo, this.dUc, this.dbi, this.dbf, this.dbo, this.dbc, this.dV, this.db, this.dWq, this.dWk, this.dWv, this.dWe, this.dbe);
        }
    }
}
