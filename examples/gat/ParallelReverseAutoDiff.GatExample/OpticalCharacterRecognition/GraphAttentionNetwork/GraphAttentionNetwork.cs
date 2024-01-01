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
                    .AddModelElementGroup("AdjacencyMatrix", new[] { numNodes, numNodes }, InitializationType.HeAdjacency)
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
            var weightMatrix = this.embeddingLayer.WeightMatrix("Embeddings");
            var gradientMatrix = this.embeddingLayer.GradientMatrix("Embeddings");
            var outputDeepMatrix = this.outputLayer.WeightDeepMatrix("DM");
            var outputGradientDeepMatrix = this.outputLayer.GradientDeepMatrix("DM");
            var outputBias = this.outputLayer.WeightMatrix("KO");
            var outputGradientBias = this.outputLayer.GradientMatrix("KO");

            List<Matrix> keys = new List<Matrix>();
            List<Matrix> keysBias = new List<Matrix>();
            List<Matrix> values = new List<Matrix>();
            List<Matrix> valuesBias = new List<Matrix>();
            List<Matrix> queries = new List<Matrix>();
            List<Matrix> queriesBias = new List<Matrix>();
            List<Matrix> reduce = new List<Matrix>();
            List<Matrix> reduceBias = new List<Matrix>();
            List<DeepMatrix> fully = new List<DeepMatrix>();
            List<DeepMatrix> fullyBias = new List<DeepMatrix>();
            List<DeepMatrix> fully2 = new List<DeepMatrix>();
            List<DeepMatrix> fully2Bias = new List<DeepMatrix>();
            List<Matrix> g = new List<Matrix>();
            List<Matrix> cs = new List<Matrix>();
            List<DeepMatrix> beta = new List<DeepMatrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                keys.Add(this.inputLayers[i].WeightMatrix("Keys"));
                keysBias.Add(this.inputLayers[i].WeightMatrix("KB"));
                values.Add(this.inputLayers[i].WeightMatrix("Values"));
                valuesBias.Add(this.inputLayers[i].WeightMatrix("VB"));
                queries.Add(this.nestedLayers[i].WeightMatrix("Queries"));
                queriesBias.Add(this.nestedLayers[i].WeightMatrix("QB"));
                reduce.Add(this.outputLayers[i].WeightMatrix("R"));
                reduceBias.Add(this.outputLayers[i].WeightMatrix("RB"));
                fully.Add(this.outputLayers[i].WeightDeepMatrix("FW"));
                fullyBias.Add(this.outputLayers[i].WeightDeepMatrix("FB"));
                fully2.Add(this.outputLayers[i].WeightDeepMatrix("F2W"));
                fully2Bias.Add(this.outputLayers[i].WeightDeepMatrix("F2B"));
                beta.Add(this.outputLayers[i].WeightDeepMatrix("Beta"));
                g.Add(this.outputLayers[i].WeightMatrix("G"));
                cs.Add(this.outputLayers[i].WeightMatrix("CS"));
            }

            List<Matrix> keysGradient = new List<Matrix>();
            List<Matrix> keysBiasGradient = new List<Matrix>();
            List<Matrix> valuesGradient = new List<Matrix>();
            List<Matrix> valuesBiasGradient = new List<Matrix>();
            List<Matrix> queriesGradient = new List<Matrix>();
            List<Matrix> queriesBiasGradient = new List<Matrix>();
            List<Matrix> reduceGradient = new List<Matrix>();
            List<Matrix> reduceBiasGradient = new List<Matrix>();
            List<DeepMatrix> fullyGradient = new List<DeepMatrix>();
            List<DeepMatrix> fullyBiasGradient = new List<DeepMatrix>();
            List<DeepMatrix> fully2Gradient = new List<DeepMatrix>();
            List<DeepMatrix> fully2BiasGradient = new List<DeepMatrix>();
            List<Matrix> gGradient = new List<Matrix>();
            List<Matrix> csGradient = new List<Matrix>();
            List<DeepMatrix> betaGradient = new List<DeepMatrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                keysGradient.Add(this.inputLayers[i].GradientMatrix("Keys"));
                keysBiasGradient.Add(this.inputLayers[i].GradientMatrix("KB"));
                valuesGradient.Add(this.inputLayers[i].GradientMatrix("Values"));
                valuesBiasGradient.Add(this.inputLayers[i].GradientMatrix("VB"));
                queriesGradient.Add(this.nestedLayers[i].GradientMatrix("Queries"));
                queriesBiasGradient.Add(this.nestedLayers[i].GradientMatrix("QB"));
                reduceGradient.Add(this.outputLayers[i].GradientMatrix("R"));
                reduceBiasGradient.Add(this.outputLayers[i].GradientMatrix("RB"));
                fullyGradient.Add(this.outputLayers[i].GradientDeepMatrix("FW"));
                fullyBiasGradient.Add(this.outputLayers[i].GradientDeepMatrix("FB"));
                fully2Gradient.Add(this.outputLayers[i].GradientDeepMatrix("F2W"));
                fully2BiasGradient.Add(this.outputLayers[i].GradientDeepMatrix("F2B"));
                gGradient.Add(this.outputLayers[i].GradientMatrix("G"));
                csGradient.Add(this.outputLayers[i].GradientMatrix("CS"));
                betaGradient.Add(this.outputLayers[i].GradientDeepMatrix("Beta"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<NestedLayersJsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new GraphAttentionComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("Input", _ => this.Input)
                .AddScalar("Divisor", x => 1d / Math.Pow(this.NumFeatures, 2))
                .AddWeight("DM", _ => outputDeepMatrix).AddGradient("DDM", _ => outputGradientDeepMatrix)
                .AddBias("KO", _ => outputBias).AddGradient("DKO", _ => outputGradientBias)
                .AddWeight("Embeddings", _ => weightMatrix).AddGradient("DEmbeddings", _ => gradientMatrix)
                .AddWeight("Keys", x => keys[x.Layer]).AddGradient("DKeys", x => keysGradient[x.Layer])
                .AddBias("KB", x => keysBias[x.Layer]).AddGradient("DKB", x => keysBiasGradient[x.Layer])
                .AddWeight("Values", x => values[x.Layer]).AddGradient("DValues", x => valuesGradient[x.Layer])
                .AddBias("VB", x => valuesBias[x.Layer]).AddGradient("DVB", x => valuesBiasGradient[x.Layer])
                .AddWeight("Queries", x => queries[x.Layer]).AddGradient("DQueries", x => queriesGradient[x.Layer])
                .AddBias("QB", x => queriesBias[x.Layer]).AddGradient("DQB", x => queriesBiasGradient[x.Layer])
                .AddWeight("R", x => reduce[x.Layer]).AddGradient("DR", x => reduceGradient[x.Layer])
                .AddBias("RB", x => reduceBias[x.Layer]).AddGradient("DRB", x => reduceBiasGradient[x.Layer])
                .AddWeight("FW", x => fully[x.Layer][x.NestedLayer]).AddGradient("DFW", x => fullyGradient[x.Layer][x.NestedLayer])
                .AddWeight("FB", x => fullyBias[x.Layer][x.NestedLayer]).AddGradient("DFB", x => fullyBiasGradient[x.Layer][x.NestedLayer])
                .AddWeight("F2W", x => fully2[x.Layer][x.NestedLayer]).AddGradient("DF2W", x => fully2Gradient[x.Layer][x.NestedLayer])
                .AddWeight("F2B", x => fully2Bias[x.Layer][x.NestedLayer]).AddGradient("DF2B", x => fully2BiasGradient[x.Layer][x.NestedLayer])
                .AddWeight("Beta", x => beta[x.Layer][x.NestedLayer]).AddGradient("DBeta", x => betaGradient[x.Layer][x.NestedLayer])
                .AddBias("G", x => g[x.Layer]).AddGradient("DG", x => gGradient[x.Layer])
                .AddBias("CS", x => cs[x.Layer]).AddGradient("DCS", x => csGradient[x.Layer])
                .AddOperationFinder("nodeFeatures", x => x.Layer == 0 ? this.computationGraph["nodeFeatures_concatenate_0_0"] : this.computationGraph[$"output_act_0_{x.Layer - 1}"])
                .AddOperationFinder("attention_scores_swiglu", x => x.NestedLayer == 0 ? this.computationGraph[$"attention_scores_0_{x.Layer}"] : this.computationGraph[$"swiglu_act_0_{x.Layer}_{x.NestedLayer - 1}"])
                .AddOperationFinder("output_act_array", x => this.computationGraph.ToOperationArray("output_act", new LayerInfo(0, 0), new LayerInfo(0, this.NumLayers - 1)))
                .AddOperationFinder("swiglu_act_array", x => this.computationGraph.ToOperationArray("swiglu_act", new LayerInfo(0, x.Layer, 0), new LayerInfo(0, x.Layer, this.NumQueries - 1)))
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
