// ------------------------------------------------------------------------------
// <copyright file="EdgeAttentionNeuralNetwork.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.EdgeAttention
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An edge attention neural network.
    /// </summary>
    public class EdgeAttentionNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.GnnExample.GraphAttentionPaths.EdgeAttention.Architecture";
        private const string ARCHITECTURE = "EdgeAttention";

        private readonly List<IModelLayer> inputLayers;
        private readonly List<IModelLayer> nestedLayers;
        private readonly List<IModelLayer> outputLayers;

        private EdgeAttentionComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="EdgeAttentionNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="numPaths">The number of paths.</param>
        /// <param name="numFeatures">The number of features.</param>
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
                outputFeaturesList.Add(numNestedFeatures * numQueries);
            }

            this.outputLayers = new List<IModelLayer>();
            numNestedFeatures = numFeatures * 2;
            for (int i = 0; i < numLayers; ++i)
            {
                var outputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("R", new[] { outputFeaturesList[i], numNestedFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("RB", new[] { 1, numNestedFeatures }, InitializationType.Zeroes);
                var outputLayer = outputLayerBuilder.Build();
                this.outputLayers.Add(outputLayer);
                numNestedFeatures *= 2;
            }

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
        /// Gets the model layers.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return this.inputLayers.Concat(this.nestedLayers).Concat(this.outputLayers);
            }
        }

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

        /// <summary>
        /// Store the operation intermediates.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void StoreOperationIntermediates(Guid id)
        {
            this.computationGraph.StoreOperationIntermediates(id);
        }

        /// <summary>
        /// Restore the operation intermediates.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void RestoreOperationIntermediates(Guid id)
        {
            this.computationGraph.RestoreOperationIntermediates(id);
        }

        /// <summary>
        /// The forward pass of the edge attention neural network.
        /// </summary>
        /// <param name="input">The input.</param>
        public void AutomaticForwardPropagate(DeepMatrix input)
        {
            // Initialize hidden state, gradients, biases, and intermediates
            this.ClearState();

            CommonMatrixUtils.SetInPlaceReplace(this.Input, input);
            var op = this.computationGraph.StartOperation;
            if (op == null)
            {
                throw new Exception("Start operation should not be null.");
            }

            IOperationBase? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);
                if (op.Id == "concatenated")
                {
                    var objArray = parameters[0] as object[] ?? throw new InvalidOperationException("Array should not be null.");
                    DeepMatrix[] deepMatrixArray = new DeepMatrix[objArray.Length];
                    for (int i = 0; i < objArray.Length; ++i)
                    {
                        var obj = objArray[i];
                        if (obj is DeepMatrix m)
                        {
                            deepMatrixArray[i] = m;
                        }
                    }

                    parameters[0] = CommonMatrixUtils.SwitchFirstTwoDimensions(deepMatrixArray);
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
        /// The backward pass of the edge attention neural network.
        /// </summary>
        /// <param name="gradient">The gradient of the loss.</param>
        /// <returns>The gradient.</returns>
        public async Task<DeepMatrix> AutomaticBackwardPropagate(DeepMatrix gradient)
        {
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_avg_0_0"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = false;
                await opVisitor.TraverseAsync();
                if (opVisitor.AggregateException != null)
                {
                    throw opVisitor.AggregateException;
                }

                opVisitor.Reset();
            }

            IOperationBase? backwardEndOperation = this.computationGraph["keys_edgeFeatures_0_0"];
            return backwardEndOperation.CalculatedGradient[0] as DeepMatrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initialize the state of the edge attention neural network.
        /// </summary>
        public void InitializeState()
        {
            if (this.Output == null)
            {
                this.Output = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumFeatures * this.NumQueries, 1));
            }
            else
            {
                this.Output.Replace(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumFeatures * this.NumQueries, 1));
            }

            if (this.Input == null)
            {
                this.Input = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumPaths, this.NumFeatures));
            }
            else
            {
                this.Input.Replace(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumPaths, this.NumFeatures));
            }
        }

        /// <summary>
        /// Clear the state of the edge attention neural network.
        /// </summary>
        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.inputLayers.ToArray());
            clearer.Clear(this.nestedLayers.ToArray());
            clearer.Clear(this.outputLayers.ToArray());
        }

        /// <summary>
        /// Initialize the computation graph of the edge attention neural network.
        /// </summary>
        /// <returns>A task.</returns>
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
            this.computationGraph = new EdgeAttentionComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddScalar("Divisor", x => 1d / Math.Pow(this.NumPaths, 2))
                .AddWeight("Keys", x => keys[x.Layer]).AddGradient("DKeys", x => keysGradient[x.Layer])
                .AddBias("KB", x => keysBias[x.Layer]).AddGradient("DKB", x => keysBiasGradient[x.Layer])
                .AddWeight("Values", x => values[x.Layer]).AddGradient("DValues", x => valuesGradient[x.Layer])
                .AddBias("VB", x => valuesBias[x.Layer]).AddGradient("DVB", x => valuesBiasGradient[x.Layer])
                .AddWeight("Queries", x => queries[x.Layer][x.NestedLayer]).AddGradient("DQueries", x => queriesGradient[x.Layer][x.NestedLayer])
                .AddBias("QB", x => queriesBias[x.Layer][x.NestedLayer]).AddGradient("DQB", x => queriesBiasGradient[x.Layer][x.NestedLayer])
                .AddWeight("R", x => reduce[x.Layer]).AddGradient("DR", x => reduceGradient[x.Layer])
                .AddBias("RB", x => reduceBias[x.Layer]).AddGradient("DRB", x => reduceBiasGradient[x.Layer])
                .AddOperationFinder("output_act_last", _ => this.computationGraph[$"output_act_0_{this.NumLayers - 1}"])
                .AddOperationFinder("edgeFeatures", x => x.Layer == 0 ? this.Input : this.computationGraph[$"output_act_0_{x.Layer - 1}"])
                .AddOperationFinder("attention_weights_values_array", x => this.computationGraph.ToOperationArray("attention_weights_values", new LayerInfo(0, x.Layer, 0), new LayerInfo(0, x.Layer, this.NumLayers - 1)))
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers, this.NumQueries);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_avg_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
