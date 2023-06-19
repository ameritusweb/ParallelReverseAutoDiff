using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention
{
    /// <summary>
    /// An edge attention neural network.
    /// </summary>
    public partial class EdgeAttentionNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention.Architecture";
        private const string ARCHITECTURE = "EdgeAttention";

        private EdgeAttentionComputationGraph computationGraph;

        private readonly List<IModelLayer> inputLayers;
        private readonly List<IModelLayer> nestedLayers;
        private readonly List<IModelLayer> outputLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="EdgeAttentionNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public EdgeAttentionNeuralNetwork(int numLayers, int numQueries, int numPaths, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumQueries = numQueries;
            this.NumPaths = numPaths;
            this.NumFeatures = numFeatures;

            this.inputLayers = new List<IModelLayer>();
            int numInputFeatures = numFeatures;
            int numInputOutputFeatures = numFeatures * 2;
            for (int i = 0; i < numLayers; ++i)
            {
                var inputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("Keys", new[] { numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("KB", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes)
                    .AddModelElementGroup("Values", new[] { numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("VB", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes);
                var inputLayer = inputLayerBuilder.Build();
                this.inputLayers.Add(inputLayer);
                numInputFeatures = numInputOutputFeatures;
                numInputOutputFeatures = numInputFeatures * 2;
            }

            this.nestedLayers = new List<IModelLayer>();
            int numNestedFeatures = numFeatures;
            int numNestedOutputFeatures = numFeatures * 2;
            List<int> outputFeaturesList = new List<int>();
            for (int i = 0; i < numLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("Queries", new[] { numQueries, numNestedFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("QB", new[] { numQueries, 1, numNestedOutputFeatures }, InitializationType.Zeroes);
                var nestedLayer = nestedLayerBuilder.Build();
                this.nestedLayers.Add(nestedLayer);
                numNestedFeatures = numNestedOutputFeatures;
                numNestedOutputFeatures = numNestedFeatures * 2;
                outputFeaturesList.Add(numNestedOutputFeatures);
            }

            this.outputLayers = new List<IModelLayer>();
            for (int i = 0; i < numLayers; ++i)
            {
                var outputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("R", new[] { outputFeaturesList[i], numFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("RB", new[] { 1, numFeatures }, InitializationType.Zeroes);
                var outputLayer = outputLayerBuilder.Build();
                this.outputLayers.Add(outputLayer);
            }

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
        /// Gets the number of queries of the neural network.
        /// </summary>
        internal int NumQueries { get; private set; }

        /// <summary>
        /// Gets the number of features of the neural network.
        /// </summary>
        internal int NumFeatures { get; private set; }

        /// <summary>
        /// Gets the number of paths of the neural network.
        /// </summary>
        internal int NumPaths { get; private set; }

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
            List<Matrix> keys = new List<Matrix>();
            List<Matrix> keysBias = new List<Matrix>();
            List<Matrix> values = new List<Matrix>();
            List<Matrix> valuesBias = new List<Matrix>();
            List<DeepMatrix> queries = new List<DeepMatrix>();
            List<DeepMatrix> queriesBias = new List<DeepMatrix>();
            List<Matrix> reduce = new List<Matrix>();
            List<Matrix> reduceBias = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                keys.Add(this.inputLayers[i].WeightMatrix("Keys"));
                keysBias.Add(this.inputLayers[i].WeightMatrix("KB"));
                values.Add(this.inputLayers[i].WeightMatrix("Values"));
                valuesBias.Add(this.inputLayers[i].WeightMatrix("VB"));
                queries.Add(this.nestedLayers[i].WeightDeepMatrix("Queries"));
                queriesBias.Add(this.nestedLayers[i].WeightDeepMatrix("QB"));
                reduce.Add(this.outputLayers[i].WeightMatrix("R"));
                reduceBias.Add(this.outputLayers[i].WeightMatrix("RB"));
            }

            List<Matrix> keysGradient = new List<Matrix>();
            List<Matrix> keysBiasGradient = new List<Matrix>();
            List<Matrix> valuesGradient = new List<Matrix>();
            List<Matrix> valuesBiasGradient = new List<Matrix>();
            List<DeepMatrix> queriesGradient = new List<DeepMatrix>();
            List<DeepMatrix> queriesBiasGradient = new List<DeepMatrix>();
            List<Matrix> reduceGradient = new List<Matrix>();
            List<Matrix> reduceBiasGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                keys.Add(this.inputLayers[i].GradientMatrix("Keys"));
                keysBias.Add(this.inputLayers[i].GradientMatrix("KB"));
                values.Add(this.inputLayers[i].GradientMatrix("Values"));
                valuesBias.Add(this.inputLayers[i].GradientMatrix("VB"));
                queries.Add(this.nestedLayers[i].GradientDeepMatrix("Queries"));
                queriesBias.Add(this.nestedLayers[i].GradientDeepMatrix("QB"));
                reduce.Add(this.outputLayers[i].GradientMatrix("R"));
                reduceBias.Add(this.outputLayers[i].GradientMatrix("RB"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<NestedLayersJsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new EdgeAttentionComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddWeight("Keys", x => keys[x.Layer])
                .AddBias("KB", x => keysBias[x.Layer])
                .AddWeight("Values", x => values[x.Layer])
                .AddBias("VB", x => valuesBias[x.Layer])
                .AddWeight("Queries", x => queries[x.Layer][x.NestedLayer])
                .AddBias("QB", x => queriesBias[x.Layer][x.NestedLayer])
                .AddWeight("R", x => reduce[x.Layer])
                .AddBias("RB", x => reduceBias[x.Layer])
                .AddOperationFinder("edgeFeatures", x => x.Layer == 0 ? this.Input : this.computationGraph[$"output_act_0_{x.Layer - 1}"])
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers, this.NumQueries);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_avg_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        public void AutomaticForwardPropagate(Matrix input)
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

            // await this.AutomaticBackwardPropagate(doNotUpdate);
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
            clearer.Clear(this.inputLayers.ToArray());
            clearer.Clear(this.nestedLayers.ToArray());
            clearer.Clear(this.outputLayers.ToArray());

            // Clear intermediates
            this.Output = CommonMatrixUtils.InitializeZeroMatrix(this.NumFeatures, 1);
            this.Input = CommonMatrixUtils.InitializeZeroMatrix(this.NumPaths, this.NumFeatures);
        }
    }
}
