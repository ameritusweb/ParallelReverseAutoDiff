// ------------------------------------------------------------------------------
// <copyright file="TransformerNeuralNetwork.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.Transformer
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.Test.Common;

    /// <summary>
    /// A transformer neural network.
    /// </summary>
    public class TransformerNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.Transformer.Architecture";
        private const string ARCHITECTURE = "Transformer";

        private readonly IModelLayer embeddingLayer;
        private readonly List<IModelLayer> inputLayers;
        private readonly List<IModelLayer> nestedLayers;
        private readonly List<IModelLayer> outputLayers;

        private TransformerComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="numPaths">The number of paths.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="alphabetSize">The alphabet size.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public TransformerNeuralNetwork(int numLayers, int numQueries, int numPaths, int numFeatures, int alphabetSize, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumQueries = numQueries;
            this.NumPaths = numPaths;
            this.NumFeatures = numFeatures;
            this.AlphabetSize = alphabetSize;

            this.embeddingLayer = new ModelLayerBuilder(this)
                .AddModelElementGroup("Embeddings", new[] { alphabetSize, numFeatures }, InitializationType.Xavier)
                .Build();

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
                numInputFeatures = numInputOutputFeatures;
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
                numNestedFeatures = numNestedOutputFeatures;
                outputFeaturesList.Add(numNestedOutputFeatures * numQueries);
            }

            this.outputLayers = new List<IModelLayer>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var outputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("FW", new[] { outputFeaturesList[i], (outputFeaturesList[i] / 2) }, InitializationType.Xavier)
                    .AddModelElementGroup("FB", new[] { 1, (outputFeaturesList[i] / 2) }, InitializationType.Xavier)
                    .AddModelElementGroup("F2W", new[] { outputFeaturesList[i], (outputFeaturesList[i] / 2) }, InitializationType.Xavier)
                    .AddModelElementGroup("F2B", new[] { 1, (outputFeaturesList[i] / 2) }, InitializationType.Xavier)
                    .AddModelElementGroup("Beta", new[] { 1, 1 }, InitializationType.He);

                    outputLayerBuilder
                        .AddModelElementGroup("R", new[] { (outputFeaturesList[i] / 2), this.NumFeatures * 2 }, InitializationType.Xavier)
                        .AddModelElementGroup("RB", new[] { 1, this.NumFeatures * 2 }, InitializationType.Zeroes);

                var outputLayer = outputLayerBuilder.Build();
                this.outputLayers.Add(outputLayer);
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
        /// Gets the position indices matrix.
        /// </summary>
        public DeepMatrix PositionIndices { get; private set; }

        /// <summary>
        /// Gets the model layers of the neural network.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return this.inputLayers.Concat(this.nestedLayers).Concat(this.outputLayers).Append(this.embeddingLayer);
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
        /// Gets the alphabet size of the neural network.
        /// </summary>
        internal int AlphabetSize { get; private set; }

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
        /// The forward pass of the readout neural network.
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
        /// The backward pass of the readout neural network.
        /// </summary>
        /// <param name="gradient">The gradient of the loss.</param>
        /// <returns>The gradient.</returns>
        public async Task<DeepMatrix> AutomaticBackwardPropagate(DeepMatrix gradient)
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
            return backwardEndOperation.CalculatedGradient[0] as DeepMatrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initializes the state of the readout neural network.
        /// </summary>
        public void InitializeState()
        {
            // Clear intermediates
            var output = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumFeatures * 2, 1));
            var input = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumPaths, this.NumFeatures));
            var positionIndices = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.AlphabetSize, 1));

            for (int i = 0; i < this.Parameters.BatchSize; ++i)
            {
                for (int j = 0; j < this.AlphabetSize; ++j)
                {
                    positionIndices[i][j][0] = j;
                }
            }

            if (this.PositionIndices == null)
            {
                this.PositionIndices = positionIndices;
            }
            else
            {
                this.PositionIndices.Replace(positionIndices.ToArray());
            }

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
        }

        /// <summary>
        /// Clears the state of the neural network.
        /// </summary>
        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.ModelLayers.ToArray());
        }

        /// <summary>
        /// Initializes the computation graph of the neural network.
        /// </summary>
        /// <returns>A task.</returns>
        private async Task InitializeComputationGraph()
        {
            var weightMatrix = this.embeddingLayer.WeightMatrix("Embeddings");
            var gradientMatrix = this.embeddingLayer.GradientMatrix("Embeddings");

            List<Matrix> keys = new List<Matrix>();
            List<Matrix> keysBias = new List<Matrix>();
            List<Matrix> values = new List<Matrix>();
            List<Matrix> valuesBias = new List<Matrix>();
            List<DeepMatrix> queries = new List<DeepMatrix>();
            List<DeepMatrix> queriesBias = new List<DeepMatrix>();
            List<Matrix> reduce = new List<Matrix>();
            List<Matrix> reduceBias = new List<Matrix>();
            List<Matrix> fully = new List<Matrix>();
            List<Matrix> fullyBias = new List<Matrix>();
            List<Matrix> fully2 = new List<Matrix>();
            List<Matrix> fully2Bias = new List<Matrix>();
            List<Matrix> beta = new List<Matrix>();
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
                fully.Add(this.outputLayers[i].WeightMatrix("FW"));
                fullyBias.Add(this.outputLayers[i].WeightMatrix("FB"));
                fully2.Add(this.outputLayers[i].WeightMatrix("F2W"));
                fully2Bias.Add(this.outputLayers[i].WeightMatrix("F2B"));
                beta.Add(this.outputLayers[i].WeightMatrix("Beta"));
            }

            List<Matrix> keysGradient = new List<Matrix>();
            List<Matrix> keysBiasGradient = new List<Matrix>();
            List<Matrix> valuesGradient = new List<Matrix>();
            List<Matrix> valuesBiasGradient = new List<Matrix>();
            List<DeepMatrix> queriesGradient = new List<DeepMatrix>();
            List<DeepMatrix> queriesBiasGradient = new List<DeepMatrix>();
            List<Matrix> reduceGradient = new List<Matrix>();
            List<Matrix> reduceBiasGradient = new List<Matrix>();
            List<Matrix> fullyGradient = new List<Matrix>();
            List<Matrix> fullyBiasGradient = new List<Matrix>();
            List<Matrix> fully2Gradient = new List<Matrix>();
            List<Matrix> fully2BiasGradient = new List<Matrix>();
            List<Matrix> betaGradient = new List<Matrix>();
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
                fullyGradient.Add(this.outputLayers[i].GradientMatrix("FW"));
                fullyBiasGradient.Add(this.outputLayers[i].GradientMatrix("FB"));
                fully2Gradient.Add(this.outputLayers[i].GradientMatrix("F2W"));
                fully2BiasGradient.Add(this.outputLayers[i].GradientMatrix("F2B"));
                betaGradient.Add(this.outputLayers[i].GradientMatrix("Beta"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<NestedLayersJsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new TransformerComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("Input", _ => this.Input)
                .AddIntermediate("PositionIndices", _ => this.PositionIndices)
                .AddScalar("Divisor", x => 1d / Math.Pow(this.NumPaths, 2))
                .AddWeight("Embeddings", _ => weightMatrix).AddGradient("DEmbeddings", _ => gradientMatrix)
                .AddWeight("Keys", x => keys[x.Layer]).AddGradient("DKeys", x => keysGradient[x.Layer])
                .AddBias("KB", x => keysBias[x.Layer]).AddGradient("DKB", x => keysBiasGradient[x.Layer])
                .AddWeight("Values", x => values[x.Layer]).AddGradient("DValues", x => valuesGradient[x.Layer])
                .AddBias("VB", x => valuesBias[x.Layer]).AddGradient("DVB", x => valuesBiasGradient[x.Layer])
                .AddWeight("Queries", x => queries[x.Layer][x.NestedLayer]).AddGradient("DQueries", x => queriesGradient[x.Layer][x.NestedLayer])
                .AddBias("QB", x => queriesBias[x.Layer][x.NestedLayer]).AddGradient("DQB", x => queriesBiasGradient[x.Layer][x.NestedLayer])
                .AddWeight("R", x => reduce[x.Layer]).AddGradient("DR", x => reduceGradient[x.Layer])
                .AddBias("RB", x => reduceBias[x.Layer]).AddGradient("DRB", x => reduceBiasGradient[x.Layer])
                .AddWeight("FW", x => fully[x.Layer]).AddGradient("DFW", x => fullyGradient[x.Layer])
                .AddBias("FB", x => fullyBias[x.Layer]).AddGradient("DFB", x => fullyBiasGradient[x.Layer])
                .AddWeight("F2W", x => fully2[x.Layer]).AddGradient("DF2W", x => fully2Gradient[x.Layer])
                .AddBias("F2B", x => fully2Bias[x.Layer]).AddGradient("DF2B", x => fully2BiasGradient[x.Layer])
                .AddBias("Beta", x => beta[x.Layer]).AddGradient("DBeta", x => betaGradient[x.Layer])
                .AddOperationFinder("output_act_last", _ => this.computationGraph[$"output_act_0_{this.NumLayers - 1}"])
                .AddOperationFinder("pathFeatures", x => x.Layer == 0 ? this.computationGraph["position_embeddings_add_0_0"] : this.computationGraph[$"output_act_0_{x.Layer - 1}"])
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
