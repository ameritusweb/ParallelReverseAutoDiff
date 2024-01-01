// ------------------------------------------------------------------------------
// <copyright file="GraphAttentionNetwork.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition.GraphAttentionNetwork
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GatExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A graph attention neural network.
    /// </summary>
    public class GraphAttentionNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition.GraphAttentionNetwork.Architecture";
        private const string ARCHITECTURE = "gat";

        private readonly List<IModelLayer> inputLayers;
        private readonly List<IModelLayer> nestedLayers;
        private readonly List<IModelLayer> outputLayers;

        private GraphAttentionComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="GraphAttentionNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public GraphAttentionNetwork(int numLayers, int numNodes, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumNodes = numNodes;
            this.NumFeatures = numFeatures;

            this.inputLayers = new List<IModelLayer>();
            int numInputOutputFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var inputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("LinearWeights", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("TransformationBias", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes)
                    .AddModelElementGroup("G", new[] { numNodes, numInputOutputFeatures }, InitializationType.He)
                    .AddModelElementGroup("Keys", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("Values", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("Queries", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier);
                var inputLayer = inputLayerBuilder.Build();
                this.inputLayers.Add(inputLayer);
                numInputOutputFeatures = numInputOutputFeatures * 2;
            }

            this.nestedLayers = new List<IModelLayer>();
            int numNestedOutputFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("AdjacencyMatrix", new[] { numLayers, numNodes, numNodes }, InitializationType.HeAdjacency)
                    .AddModelElementGroup("AttentionWeights", new[] { 1, numNestedOutputFeatures * 2 }, InitializationType.Xavier);
                var nestedLayer = nestedLayerBuilder.Build();
                this.nestedLayers.Add(nestedLayer);
                numNestedOutputFeatures = numNestedOutputFeatures * 2;
            }

            this.outputLayers = new List<IModelLayer>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var outputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("FW", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("FB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("F2W", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("F2B", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("Beta", new[] { 1, 1 }, InitializationType.He);
                var outputLayer = outputLayerBuilder.Build();
                this.outputLayers.Add(outputLayer);
                numNestedOutputFeatures = numNestedOutputFeatures * 2;
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
        /// Gets the model layers of the neural network.
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
        /// Gets the number of features of the neural network.
        /// </summary>
        internal int NumFeatures { get; private set; }

        /// <summary>
        /// Gets the number of nodes of the neural network.
        /// </summary>
        internal int NumNodes { get; private set; }

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
        public void AutomaticForwardPropagate(Matrix input)
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

                if (op.Id == "deep_concatenate")
                {
                    var objArray = parameters[0] as object[] ?? throw new InvalidOperationException("Array should not be null.");
                    Matrix[] matrixArray = new Matrix[objArray.Length];
                    for (int i = 0; i < objArray.Length; ++i)
                    {
                        var obj = objArray[i];
                        if (obj is Matrix m)
                        {
                            matrixArray[i] = m;
                        }
                    }

                    parameters[0] = new DeepMatrix(matrixArray);
                }

                var forward = op.OperationType.GetMethod("Forward", parameters.Select(x => x.GetType()).ToArray());
                if (forward == null)
                {
                    throw new Exception($"Forward method not found for operation {op.OperationType.Name}");
                }

                forward.Invoke(op, parameters);
                var output = op.GetOutput();
                var deepOutput = op.GetDeepOutput();
                if (output != null)
                {
                    if (double.IsNaN(output[0][0]))
                    {
                    }
                }
                else if (deepOutput != null)
                {
                    if (double.IsNaN(deepOutput[0][0][0]))
                    {
                    }
                }

                if (op.ResultToName != null)
                {
                    var split = op.ResultToName.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                    var oo = this.computationGraph[MatrixType.Intermediate, split[0], op.LayerInfo];
                    try
                    {
                        op.CopyResult(oo);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                    }
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
        public async Task<Matrix> AutomaticBackwardPropagate(Matrix gradient)
        {
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_0_0"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = true;
                await opVisitor.TraverseAsync();
                if (opVisitor.AggregateException != null)
                {
                    if (opVisitor.AggregateException.InnerExceptions.Count > 1)
                    {
                        throw opVisitor.AggregateException;
                    }
                    else
                    {
                        Console.WriteLine(opVisitor.AggregateException.InnerExceptions[0].Message);
                    }
                }

                opVisitor.Reset();
            }

            IOperationBase? backwardEndOperation = this.computationGraph["node_features_transform_0_0"];
            if (backwardEndOperation.CalculatedGradient == null)
            {
                return gradient;
            }

            return backwardEndOperation.CalculatedGradient[1] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initialize the state of the edge attention neural network.
        /// </summary>
        public void InitializeState()
        {
            // Clear intermediates
            var output = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(this.NumNodes, this.NumFeatures).ToArray());
            var input = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(this.NumNodes, this.NumFeatures).ToArray());

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
        /// Clear the state of the edge attention neural network.
        /// </summary>
        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.ModelLayers.ToArray());
        }

        /// <summary>
        /// Initialize the computation graph of the edge attention neural network.
        /// </summary>
        /// <returns>A task.</returns>
        private async Task InitializeComputationGraph()
        {
            List<Matrix> linearWeights = new List<Matrix>();
            List<Matrix> transformationBias = new List<Matrix>();
            List<Matrix> keys = new List<Matrix>();
            List<Matrix> values = new List<Matrix>();
            List<Matrix> queries = new List<Matrix>();
            List<DeepMatrix> adjacency = new List<DeepMatrix>();
            List<Matrix> attentionWeights = new List<Matrix>();
            List<Matrix> fully = new List<Matrix>();
            List<Matrix> fullyBias = new List<Matrix>();
            List<Matrix> fully2 = new List<Matrix>();
            List<Matrix> fully2Bias = new List<Matrix>();
            List<Matrix> g = new List<Matrix>();
            List<Matrix> beta = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                linearWeights.Add(this.inputLayers[i].WeightMatrix("LinearWeights"));
                transformationBias.Add(this.inputLayers[i].WeightMatrix("TransformationBias"));
                keys.Add(this.inputLayers[i].WeightMatrix("Keys"));
                values.Add(this.inputLayers[i].WeightMatrix("Values"));
                queries.Add(this.nestedLayers[i].WeightMatrix("Queries"));
                adjacency.Add(this.nestedLayers[i].WeightDeepMatrix("AdjacencyMatrix"));
                attentionWeights.Add(this.nestedLayers[i].WeightMatrix("AttentionWeights"));
                fully.Add(this.outputLayers[i].WeightMatrix("FW"));
                fullyBias.Add(this.outputLayers[i].WeightMatrix("FB"));
                fully2.Add(this.outputLayers[i].WeightMatrix("F2W"));
                fully2Bias.Add(this.outputLayers[i].WeightMatrix("F2B"));
                beta.Add(this.outputLayers[i].WeightMatrix("Beta"));
                g.Add(this.outputLayers[i].WeightMatrix("G"));
            }

            List<Matrix> linearWeightsGradient = new List<Matrix>();
            List<Matrix> transformationBiasGradient = new List<Matrix>();
            List<Matrix> keysGradient = new List<Matrix>();
            List<Matrix> valuesGradient = new List<Matrix>();
            List<Matrix> queriesGradient = new List<Matrix>();
            List<DeepMatrix> adjacencyGradient = new List<DeepMatrix>();
            List<Matrix> attentionWeightsGradient = new List<Matrix>();
            List<Matrix> fullyGradient = new List<Matrix>();
            List<Matrix> fullyBiasGradient = new List<Matrix>();
            List<Matrix> fully2Gradient = new List<Matrix>();
            List<Matrix> fully2BiasGradient = new List<Matrix>();
            List<Matrix> gGradient = new List<Matrix>();
            List<Matrix> betaGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                linearWeightsGradient.Add(this.inputLayers[i].GradientMatrix("LinearWeights"));
                transformationBiasGradient.Add(this.inputLayers[i].GradientMatrix("TransformationBias"));
                keysGradient.Add(this.inputLayers[i].GradientMatrix("Keys"));
                valuesGradient.Add(this.inputLayers[i].GradientMatrix("Values"));
                queriesGradient.Add(this.nestedLayers[i].GradientMatrix("Queries"));
                adjacencyGradient.Add(this.nestedLayers[i].GradientDeepMatrix("AdjacencyMatrix"));
                attentionWeightsGradient.Add(this.nestedLayers[i].GradientMatrix("AttentionWeights"));
                fullyGradient.Add(this.outputLayers[i].GradientMatrix("FW"));
                fullyBiasGradient.Add(this.outputLayers[i].GradientMatrix("FB"));
                fully2Gradient.Add(this.outputLayers[i].GradientMatrix("F2W"));
                fully2BiasGradient.Add(this.outputLayers[i].GradientMatrix("F2B"));
                betaGradient.Add(this.outputLayers[i].GradientMatrix("Beta"));
                gGradient.Add(this.outputLayers[i].GradientMatrix("G"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<NestedLayersJsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new GraphAttentionComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddScalar("Divisor", x => 1d / Math.Pow(this.NumFeatures, 2))
                .AddWeight("LinearWeights", x => linearWeights[x.Layer]).AddGradient("DLinearWeights", x => linearWeightsGradient[x.Layer])
                .AddWeight("TransformationBias", x => transformationBias[x.Layer]).AddGradient("DTransformationBias", x => transformationBiasGradient[x.Layer])
                .AddWeight("Keys", x => keys[x.Layer]).AddGradient("DKeys", x => keysGradient[x.Layer])
                .AddWeight("Values", x => values[x.Layer]).AddGradient("DValues", x => valuesGradient[x.Layer])
                .AddWeight("Queries", x => queries[x.Layer]).AddGradient("DQueries", x => queriesGradient[x.Layer])
                .AddWeight("FW", x => fully[x.Layer]).AddGradient("DFW", x => fullyGradient[x.Layer])
                .AddWeight("FB", x => fullyBias[x.Layer]).AddGradient("DFB", x => fullyBiasGradient[x.Layer])
                .AddWeight("F2W", x => fully2[x.Layer]).AddGradient("DF2W", x => fully2Gradient[x.Layer])
                .AddWeight("F2B", x => fully2Bias[x.Layer]).AddGradient("DF2B", x => fully2BiasGradient[x.Layer])
                .AddWeight("Beta", x => beta[x.Layer]).AddGradient("DBeta", x => betaGradient[x.Layer])
                .AddBias("G", x => g[x.Layer]).AddGradient("DG", x => gGradient[x.Layer])
                .AddWeight("AdjacencyMatrix", x => adjacency[x.Layer][x.NestedLayer]).AddGradient("DAdjacencyMatrix", x => adjacencyGradient[x.Layer][x.NestedLayer])
                .AddWeight("AttentionWeights", x => attentionWeights[x.Layer]).AddGradient("DAttentionWeights", x => attentionWeightsGradient[x.Layer])
                .AddOperationFinder("nodeFeatures", x => x.Layer == 0 ? this.Input : this.computationGraph[$"swiglu_act_0_{x.Layer - 1}"])
                .AddOperationFinder("swiglu_act_last", _ => this.computationGraph[$"swiglu_act_0_{this.NumLayers - 1}"])
                .AddOperationFinder("feature_aggregation_array", x => this.computationGraph.ToOperationArray("feature_aggregation", new LayerInfo(0, x.Layer, 0), new LayerInfo(0, x.Layer, this.NumLayers - 1)))
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
