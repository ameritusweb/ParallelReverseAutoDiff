﻿using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing
{
    /// <summary>
    /// An attention message passing neural network.
    /// </summary>
    public class AttentionMessagePassingNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing.Architecture";
        private const string ARCHITECTURE = "MessagePassing";

        private AttentionMessagePassingComputationGraph computationGraph;

        private IModelLayer hiddenLayer;

        /// <summary>
        /// Initializes a new instance of the <see cref="AttentionMessagePassingNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
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
        public Matrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public Matrix Output { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets the connected paths deep matrix.
        /// </summary>
        public DeepMatrix ConnectedPathsDeepMatrix { get; set; }

        /// <summary>
        /// Gets the connected paths deep matrix.
        /// </summary>
        public DeepMatrix DConnectedPathsDeepMatrix { get; set; }

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

        private void ClearState()
        {

        }

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
                .AddIntermediate("ConnectedPathsDeepMatrix", _ => this.ConnectedPathsDeepMatrix)
                .AddScalar("connectedPathsMatrixRows", _ => this.ConnectedPathsDeepMatrix.Rows)
                .AddScalar("connectedPathsMatrixColumns", _ => this.ConnectedPathsDeepMatrix.Cols)
                .AddGradient("DConnectedPathsDeepMatrix", x => this.DConnectedPathsDeepMatrix)
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

        public void StoreOperationIntermediates(Guid id)
        {
            this.computationGraph.StoreOperationIntermediates(id);
        }

        public void RestoreOperationIntermediates(Guid id)
        {
            this.computationGraph.RestoreOperationIntermediates(id);
        }

        public void AutomaticForwardPropagate(Matrix input, bool doNotUpdate)
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
                var forward = op.OperationType.GetMethod("Forward");
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

            //await this.AutomaticBackwardPropagate(doNotUpdate);
        }

        public async Task<Matrix> AutomaticBackwardPropagate(Matrix gradient)
        {
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph[$"current_path_features_0_{this.NumLayers - 1}"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = true;
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
            }
            IOperationBase? backwardEndOperation = this.computationGraph["weighted_currentPathFeatures_0_0"];
            return backwardEndOperation.CalculatedGradient[1] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        private void InitializeState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] { this.hiddenLayer});

            this.ConnectedPathsDeepMatrix = new DeepMatrix(this.NumLayers, this.NumPaths, this.NumFeatures);
            this.DConnectedPathsDeepMatrix = new DeepMatrix(this.NumLayers, this.NumPaths, this.NumFeatures);

            // Clear intermediates
            this.Output = CommonMatrixUtils.InitializeZeroMatrix(this.NumFeatures, 1);
            this.Input = CommonMatrixUtils.InitializeZeroMatrix(this.NumFeatures, 1);
        }
    }
}