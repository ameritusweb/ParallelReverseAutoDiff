using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN
{
    /// <summary>
    /// A readout neural network.
    /// </summary>
    public class ReadoutNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.Readout.Architecture";
        private const string ARCHITECTURE = "Readout";

        private ReadoutComputationGraph computationGraph;

        private readonly List<IModelLayer> inputLayers;
        private readonly List<IModelLayer> nestedLayers;
        private readonly List<IModelLayer> outputLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReadoutNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public ReadoutNeuralNetwork(int numLayers, int numQueries, int numPaths, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers * 10;
            this.NumQueries = numQueries;
            this.NumPaths = numPaths;
            this.NumFeatures = numFeatures * (int)Math.Pow(2, numLayers) * 8;

            this.inputLayers = new List<IModelLayer>();
            int numInputFeatures = this.NumFeatures;
            int numInputOutputFeatures = this.NumFeatures * 2;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var inputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("Keys", new[] { numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("KB", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes)
                    .AddModelElementGroup("Values", new[] { numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("VB", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes);
                var inputLayer = inputLayerBuilder.Build();
                this.inputLayers.Add(inputLayer);
            }

            this.nestedLayers = new List<IModelLayer>();
            int numNestedFeatures = this.NumFeatures;
            int numNestedOutputFeatures = this.NumFeatures * 2;
            List<int> outputFeaturesList = new List<int>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("Queries", new[] { numQueries, numNestedFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("QB", new[] { numQueries, 1, numNestedOutputFeatures }, InitializationType.Zeroes);
                var nestedLayer = nestedLayerBuilder.Build();
                this.nestedLayers.Add(nestedLayer);
                outputFeaturesList.Add(numNestedOutputFeatures * numQueries);
            }

            this.outputLayers = new List<IModelLayer>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var outputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("R", new[] { outputFeaturesList[i], this.NumFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("RB", new[] { 1, this.NumFeatures }, InitializationType.Zeroes);
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
        /// Gets the AV matrix array.
        /// </summary>
        public Matrix[] AV { get; private set; }

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
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.inputLayers.ToArray());
            clearer.Clear(this.nestedLayers.ToArray());
            clearer.Clear(this.outputLayers.ToArray());
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
                keysGradient.Add(this.inputLayers[i].GradientMatrix("Keys"));
                keysBiasGradient.Add(this.inputLayers[i].GradientMatrix("KB"));
                valuesGradient.Add(this.inputLayers[i].GradientMatrix("Values"));
                valuesBiasGradient.Add(this.inputLayers[i].GradientMatrix("VB"));
                queriesGradient.Add(this.nestedLayers[i].GradientDeepMatrix("Queries"));
                queriesBiasGradient.Add(this.nestedLayers[i].GradientDeepMatrix("QB"));
                reduceGradient.Add(this.outputLayers[i].GradientMatrix("R"));
                reduceBiasGradient.Add(this.outputLayers[i].GradientMatrix("RB"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<NestedLayersJsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new ReadoutComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("AV", x => this.AV[x.Layer])
                .AddScalar("Divisor", x => 1d / Math.Pow(this.NumFeatures, 2 + x.Layer))
                .AddWeight("Keys", x => keys[x.Layer]).AddGradient("DKeys", x => keysGradient[x.Layer])
                .AddBias("KB", x => keysBias[x.Layer]).AddGradient("DKB", x => keysBiasGradient[x.Layer])
                .AddWeight("Values", x => values[x.Layer]).AddGradient("DValues", x => valuesGradient[x.Layer])
                .AddBias("VB", x => valuesBias[x.Layer]).AddGradient("DVB", x => valuesBiasGradient[x.Layer])
                .AddWeight("Queries", x => queries[x.Layer][x.NestedLayer]).AddGradient("DQueries", x => queriesGradient[x.Layer][x.NestedLayer])
                .AddBias("QB", x => queriesBias[x.Layer][x.NestedLayer]).AddGradient("DQB", x => queriesBiasGradient[x.Layer][x.NestedLayer])
                .AddWeight("R", x => reduce[x.Layer]).AddGradient("DR", x => reduceGradient[x.Layer])
                .AddBias("RB", x => reduceBias[x.Layer]).AddGradient("DRB", x => reduceBiasGradient[x.Layer])
                .AddOperationFinder("output_act_last", _ => this.computationGraph[$"output_act_0_{this.NumLayers - 1}"])
                .AddOperationFinder("pathFeatures", x => x.Layer == 0 ? this.Input : this.computationGraph[$"output_act_0_{x.Layer - 1}"])
                .AddOperationFinder("attention_weights_values_array", x => this.computationGraph.ToOperationArray("attention_weights_values", new LayerInfo(0, x.Layer, 0), new LayerInfo(0, x.Layer, this.NumLayers - 1)))
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers, this.NumQueries);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_avg_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
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

                if (op.Id == "concatenated")
                {
                    var objArray = parameters[0] as object[] ?? throw new InvalidOperationException("Array should not be null.");
                    DeepMatrix deepMatrix = new DeepMatrix(objArray.Length, 1, 1);
                    for (int i = 0; i < objArray.Length; ++i)
                    {
                        var obj = objArray[i];
                        if (obj is Matrix m)
                        {
                            deepMatrix[i] = m;
                        }
                    }
                    parameters[0] = deepMatrix;
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

        public async Task<Matrix> AutomaticBackwardPropagate(Matrix gradient)
        {
            int traverseCount = 0;
            IOperationBase? backwardStartOperation = this.computationGraph["output_avg_0_0"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = true;
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
                traverseCount++;
            }
            IOperationBase? backwardEndOperation = this.computationGraph["keys_pathFeatures_0_0"];
            return backwardEndOperation.CalculatedGradient[0] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        private void InitializeState()
        {
            // Clear intermediates
            this.Output = CommonMatrixUtils.InitializeZeroMatrix(this.NumFeatures, 1);
            this.Input = CommonMatrixUtils.InitializeZeroMatrix(this.NumPaths, this.NumFeatures);
        }
    }
}
