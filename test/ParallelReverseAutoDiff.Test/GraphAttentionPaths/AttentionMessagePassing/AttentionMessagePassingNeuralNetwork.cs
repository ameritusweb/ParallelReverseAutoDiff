using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.AttentionMessagePassing
{
    /// <summary>
    /// An attention message passing neural network.
    /// </summary>
    public partial class AttentionMessagePassingNeuralNetwork : NeuralNetwork
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
            this.NumFeatures = numFeatures;

            var hiddenLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Weights", new[] { 6, numFeatures, numFeatures }, InitializationType.Xavier) // one weight per piece type
                .AddModelElementGroup("B", new[] { 6, numFeatures, 1 }, InitializationType.Zeroes) // similarly one bias per piece type
                .AddModelElementGroup("ConnectedWeights", new[] { numPaths, numFeatures, numFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("CB", new[] { numPaths, 1, numFeatures }, InitializationType.Zeroes);
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

            this.Parameters.AdamIteration = iterationIndex + 1;

            await this.AutomaticForwardPropagate(input, doNotUpdate.Value);
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
                .AddWeight("Weights", x => weights[x.Layer]).AddGradient("DWeights", x => weightsGradient[x.Layer])
                .AddBias("B", x => b[x.Layer]).AddGradient("DB", x => bGradient[x.Layer])
                .AddWeight("ConnectedWeights", x => connected[x.Layer]).AddGradient("DConnectedWeights", x => connectedGradient[x.Layer])
                .AddBias("CB", x => cb[x.Layer]).AddGradient("DCB", x => cbGradient[x.Layer])
                .AddOperationFinder("CurrentPathFeatures", x => x.Layer == 0 ? this.Input : this.computationGraph[$"current_path_features_0_{x.Layer - 1}"])
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_t_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        private async Task AutomaticForwardPropagate(Matrix input, bool doNotUpdate)
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
            var gradientOfLossWrtOutput = (lossFunction.Backward(this.Output).Item1 as Matrix) ?? throw new Exception("Gradient of the loss wrt the output should not be null.");
            int traverseCount = 0;
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_t_0_0"];
            if (gradientOfLossWrtOutput[0][0] != 0.0d)
            {
                backwardStartOperation.BackwardInput = gradientOfLossWrtOutput;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = true;
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
                traverseCount++;
            }

            if (traverseCount == 0 || doNotUpdate)
            {
                return;
            }
        }

        private void InitializeState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] { this.hiddenLayer});

            // Clear intermediates
            this.Output = CommonMatrixUtils.InitializeZeroMatrix(this.NumFeatures, 1);
            this.Input = CommonMatrixUtils.InitializeZeroMatrix(this.NumPaths, this.NumFeatures);
        }
    }
}
