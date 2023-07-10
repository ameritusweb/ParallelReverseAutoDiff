// ------------------------------------------------------------------------------
// <copyright file="AttentionMessagePassingNeuralNetwork.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An attention message passing neural network.
    /// </summary>
    public class AttentionMessagePassingNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.GnnExample.GraphAttentionPaths.AttentionMessagePassing.Architecture";
        private const string ARCHITECTURE = "MessagePassing";

        private AttentionMessagePassingComputationGraph computationGraph;

        private IModelLayer hiddenLayer;

        /// <summary>
        /// Initializes a new instance of the <see cref="AttentionMessagePassingNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numPaths">The number of paths.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public AttentionMessagePassingNeuralNetwork(int numLayers, int numPaths, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumPaths = numPaths;
            this.NumFeatures = numFeatures * (int)Math.Pow(2d, (int)numLayers) * 2;

            var hiddenLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Weights", new[] { numLayers, this.NumFeatures, this.NumFeatures }, InitializationType.Xavier) // one weight per piece type
                .AddModelElementGroup("B", new[] { numLayers, this.NumFeatures, 1 }, InitializationType.Zeroes) // similarly one bias per piece type
                .AddModelElementGroup("ConnectedWeights", new[] { numLayers, this.NumFeatures, this.NumFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("CB", new[] { numLayers, 1, this.NumFeatures }, InitializationType.Zeroes);
            this.hiddenLayer = hiddenLayerBuilder.Build();

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input matrix.
        /// </summary>
        public DeepMatrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public DeepMatrix Output { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets or sets the connected paths deep matrix.
        /// </summary>
        public FourDimensionalMatrix ConnectedPathsDeepMatrixArray { get; set; }

        /// <summary>
        /// Gets or sets the connected paths deep matrix.
        /// </summary>
        public FourDimensionalMatrix DConnectedPathsDeepMatrixArray { get; set; }

        /// <summary>
        /// Gets the model layers.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return new[] { this.hiddenLayer };
            }
        }

        /// <summary>
        /// Gets the number of layers of the neural network.
        /// </summary>
        internal int NumLayers { get; private set; }

        /// <summary>
        /// Gets the number of paths of the neural network.
        /// </summary>
        internal int NumPaths { get; private set; }

        /// <summary>
        /// Gets the number of features of the neural network.
        /// </summary>
        internal int NumFeatures { get; private set; }

        /// <summary>
        /// Initializes the computation graph of the convolutional neural network.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            await this.InitializeComputationGraph();
        }

        /// <summary>
        /// Stores the operation intermediates.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void StoreOperationIntermediates(Guid id)
        {
            this.computationGraph.StoreOperationIntermediates(id);
        }

        /// <summary>
        /// Restores the operation intermediates.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void RestoreOperationIntermediates(Guid id)
        {
            this.computationGraph.RestoreOperationIntermediates(id);
        }

        /// <summary>
        /// The forward pass of the attention message passing neural network.
        /// </summary>
        /// <param name="input">The input.</param>
        public void AutomaticForwardPropagate(DeepMatrix input)
        {
            // Initialize hidden state, gradients, biases, and intermediates
            this.ClearState();

            CommonMatrixUtils.SetInPlace(this.Input, input);
            var op = this.computationGraph.StartOperation;
            if (op == null)
            {
                throw new Exception("Start operation should not be null.");
            }

            IOperationBase? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);
                if (op.Id == "connected_paths_deepmatrix")
                {
                    parameters[0] = ((FourDimensionalMatrix)parameters[0]).ToArray();
                }

                var forward = op.OperationType.GetMethod("Forward", parameters.Select(x => x.GetType()).ToArray());
                if (forward == null)
                {
                    throw new Exception($"Forward method not found for operation {op.OperationType.Name}");
                }

                forward.Invoke(op, parameters);
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
        }

        /// <summary>
        /// The backward pass of the attention message passing neural network.
        /// </summary>
        /// <param name="gradient">The gradient of the loss.</param>
        /// <returns>The gradient.</returns>
        public async Task<DeepMatrix> AutomaticBackwardPropagate(DeepMatrix gradient)
        {
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph[$"current_path_features_0_{this.NumLayers - 1}"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = false;
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
            }

            IOperationBase? backwardEndOperation = this.computationGraph["weighted_currentPathFeatures_0_0"];
            return backwardEndOperation.CalculatedGradient[1] as DeepMatrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initializes the state of the attention message passing neural network.
        /// </summary>
        public void InitializeState()
        {
            var connectedPathsDeepMatrixArray = CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumLayers, this.NumPaths, this.NumFeatures).Select(x => new DeepMatrix(x)).ToArray();
            var dConnectedPathsDeepMatrixArray = CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumLayers, this.NumPaths, this.NumFeatures).Select(x => new DeepMatrix(x)).ToArray();

            // Clear intermediates
            var output = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumFeatures, 1));
            var input = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumFeatures, 1));

            if (this.Output == null)
            {
                this.Output = output;
            }
            else
            {
                this.Output.Replace(output.ToArray());
            }

            if (this.Input == null)
            {
                this.Input = input;
            }
            else
            {
                this.Input.Replace(input.ToArray());
            }

            if (this.ConnectedPathsDeepMatrixArray == null)
            {
                this.ConnectedPathsDeepMatrixArray = new FourDimensionalMatrix(connectedPathsDeepMatrixArray);
            }
            else
            {
                this.ConnectedPathsDeepMatrixArray.Replace(connectedPathsDeepMatrixArray.ToArray());
            }

            if (this.DConnectedPathsDeepMatrixArray == null)
            {
                this.DConnectedPathsDeepMatrixArray = new FourDimensionalMatrix(dConnectedPathsDeepMatrixArray);
            }
            else
            {
                this.DConnectedPathsDeepMatrixArray.Replace(dConnectedPathsDeepMatrixArray.ToArray());
            }
        }

        /// <summary>
        /// Clears the state of the attention message passing neural network.
        /// </summary>
        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] { this.hiddenLayer });
        }

        /// <summary>
        /// Initializes the computation graph of the attention message passing neural network.
        /// </summary>
        /// <returns>The task.</returns>
        private async Task InitializeComputationGraph()
        {
            var weights = this.hiddenLayer.WeightDeepMatrix("Weights");
            var b = this.hiddenLayer.WeightDeepMatrix("B");
            var connected = this.hiddenLayer.WeightDeepMatrix("ConnectedWeights");
            var cb = this.hiddenLayer.WeightDeepMatrix("CB");

            var weightsGradient = this.hiddenLayer.GradientDeepMatrix("Weights");
            var bGradient = this.hiddenLayer.GradientDeepMatrix("B");
            var connectedGradient = this.hiddenLayer.GradientDeepMatrix("ConnectedWeights");
            var cbGradient = this.hiddenLayer.GradientDeepMatrix("CB");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new AttentionMessagePassingComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("ConnectedPathsDeepMatrix", _ => this.ConnectedPathsDeepMatrixArray)
                .AddDynamic("connectedPathsMatrixRows", _ => new Matrix(this.ConnectedPathsDeepMatrixArray.Count, 1).ReplaceVertically(this.ConnectedPathsDeepMatrixArray.Select(x => (double)x.Depth).ToArray()))
                .AddDynamic("connectedPathsMatrixColumns", _ => new Matrix(this.ConnectedPathsDeepMatrixArray.Count, 1).ReplaceVertically(this.ConnectedPathsDeepMatrixArray.Select(x => (double)x.Rows).ToArray()))
                .AddGradient("DConnectedPathsDeepMatrix", x => this.DConnectedPathsDeepMatrixArray)
                .AddWeight("Weights", x => weights[x.Layer]).AddGradient("DWeights", x => weightsGradient[x.Layer])
                .AddBias("B", x => b[x.Layer]).AddGradient("DB", x => bGradient[x.Layer])
                .AddWeight("ConnectedWeights", x => connected[x.Layer]).AddGradient("DConnectedWeights", x => connectedGradient[x.Layer])
                .AddBias("CB", x => cb[x.Layer]).AddGradient("DCB", x => cbGradient[x.Layer])
                .AddOperationFinder("CurrentPathFeatures", x => x.Layer == 0 ? this.Input : this.computationGraph[$"current_path_features_0_{x.Layer - 1}"])
                .AddOperationFinder("connected_paths_matrix_trans_find", _ => this.computationGraph["connected_paths_matrix_trans_0_0"])
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph[$"current_path_features_0_{this.NumLayers - 1}"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
