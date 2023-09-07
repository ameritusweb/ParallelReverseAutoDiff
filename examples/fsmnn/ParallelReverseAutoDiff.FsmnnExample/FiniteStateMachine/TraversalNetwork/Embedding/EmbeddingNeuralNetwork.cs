// ------------------------------------------------------------------------------
// <copyright file="EmbeddingNeuralNetwork.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.TraversalNetwork.Embedding
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An embedding neural network.
    /// </summary>
    public class EmbeddingNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.FsmnnExample.FiniteStateMachine.TraversalNetwork.Embedding.Architecture";
        private const string ARCHITECTURE = "Embedding";

        private readonly IModelLayer embeddingLayer;
        private readonly List<IModelLayer> inputLayers;
        private readonly List<IModelLayer> nestedLayers;
        private readonly List<IModelLayer> outputLayers;
        private readonly IModelLayer outputLayer;

        private EmbeddingComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="EmbeddingNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numIndices">The number of indices.</param>
        /// <param name="alphabetSize">The alphabet size.</param>
        /// <param name="embeddingSize">The embedding size.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public EmbeddingNeuralNetwork(int numLayers, int numQueries, int numNodes, int numFeatures, int numIndices, int alphabetSize, int embeddingSize, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.AlphabetSize = alphabetSize;
            this.NumIndices = numIndices;
            this.EmbeddingSize = embeddingSize;
            this.NumLayers = numLayers;
            this.NumQueries = numQueries;
            this.NumNodes = numNodes;
            this.NumFeatures = numFeatures;

            this.embeddingLayer = new ModelLayerBuilder(this)
                .AddModelElementGroup("Embeddings", new[] { alphabetSize, embeddingSize }, InitializationType.Xavier)
                .Build();

            this.inputLayers = new List<IModelLayer>();
            int numInputFeatures = this.NumFeatures;
            int numInputOutputFeatures = this.NumFeatures * 2;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var inputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("Keys", new[] { i == 0 ? 1 : numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("KB", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes)
                    .AddModelElementGroup("Values", new[] { i == 0 ? 1 : numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("VB", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes);
                var inputLayer = inputLayerBuilder.Build();
                this.inputLayers.Add(inputLayer);
            }

            this.nestedLayers = new List<IModelLayer>();
            int numNestedFeatures = this.NumFeatures;
            int numNestedOutputFeatures = this.NumFeatures * 2;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("Queries", new[] { numNestedFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("QB", new[] { 1, numNestedOutputFeatures }, InitializationType.Zeroes);
                var nestedLayer = nestedLayerBuilder.Build();
                this.nestedLayers.Add(nestedLayer);
            }

            this.outputLayers = new List<IModelLayer>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var outputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("FW", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("FB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("F2W", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("F2B", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("Beta", new[] { 1, 1 }, InitializationType.He)
                    .AddModelElementGroup("R", new[] { numNestedOutputFeatures, this.NumFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("RB", new[] { 1, this.NumFeatures }, InitializationType.Zeroes)
                    .AddModelElementGroup("G", new[] { this.NumFeatures, this.NumFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("CS", new[] { this.NumFeatures, this.NumFeatures }, InitializationType.Xavier);
                var outputLayer = outputLayerBuilder.Build();
                this.outputLayers.Add(outputLayer);
            }

            this.outputLayer = new ModelLayerBuilder(this)
                .AddModelElementGroup("DM", new[] { numNodes, this.NumFeatures, this.NumFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("KO", new[] { 1, numNodes }, InitializationType.Zeroes)
                .Build();

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input matrix.
        /// </summary>
        public DeepMatrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public Matrix Output { get; private set; }

        /// <summary>
        /// Gets or sets the count of the path.
        /// </summary>
        public int NumPath { get; set; }

        /// <summary>
        /// Gets the model layers of the neural network.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return this.inputLayers.Concat(this.nestedLayers).Concat(this.outputLayers).Append(this.embeddingLayer).Append(this.outputLayer);
            }
        }

        /// <summary>
        /// Gets the alphabet size of the neural network.
        /// </summary>
        internal int AlphabetSize { get; private set; }

        /// <summary>
        /// Gets the number of indices of the neural network.
        /// </summary>
        internal int NumIndices { get; private set; }

        /// <summary>
        /// Gets the embedding size of the neural network.
        /// </summary>
        internal int EmbeddingSize { get; private set; }

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

                if (op.Id == "reduce")
                {
                }

                if (op.Id == "keys_nodeFeatures")
                {
                }

                if (op.Id == "queries_keys_transpose")
                {
                }

                if (op.Id == "fully_connected")
                {
                }

                if (op.Id == "pre_output")
                {
                }

                if (op.Id == "output_act_summation")
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

                if (op.Id == "output_norm_cosine")
                {
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
            backwardStartOperation = this.computationGraph["output_softmax_0_0"];
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

            IOperationBase? backwardEndOperation = this.computationGraph["batch_embeddings_0_0"];
            return backwardEndOperation.CalculatedGradient[1] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initialize the state of the edge attention neural network.
        /// </summary>
        public void InitializeState()
        {
            // Clear intermediates
            var output = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(1, this.NumNodes).ToArray());
            var input = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.NumPath, this.NumIndices, 1));

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
            List<Matrix> fully = new List<Matrix>();
            List<Matrix> fullyBias = new List<Matrix>();
            List<Matrix> fully2 = new List<Matrix>();
            List<Matrix> fully2Bias = new List<Matrix>();
            List<Matrix> g = new List<Matrix>();
            List<Matrix> cs = new List<Matrix>();
            List<Matrix> beta = new List<Matrix>();
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
                fully.Add(this.outputLayers[i].WeightMatrix("FW"));
                fullyBias.Add(this.outputLayers[i].WeightMatrix("FB"));
                fully2.Add(this.outputLayers[i].WeightMatrix("F2W"));
                fully2Bias.Add(this.outputLayers[i].WeightMatrix("F2B"));
                g.Add(this.outputLayers[i].WeightMatrix("G"));
                cs.Add(this.outputLayers[i].WeightMatrix("CS"));
                beta.Add(this.outputLayers[i].WeightMatrix("Beta"));
            }

            List<Matrix> keysGradient = new List<Matrix>();
            List<Matrix> keysBiasGradient = new List<Matrix>();
            List<Matrix> valuesGradient = new List<Matrix>();
            List<Matrix> valuesBiasGradient = new List<Matrix>();
            List<Matrix> queriesGradient = new List<Matrix>();
            List<Matrix> queriesBiasGradient = new List<Matrix>();
            List<Matrix> reduceGradient = new List<Matrix>();
            List<Matrix> reduceBiasGradient = new List<Matrix>();
            List<Matrix> fullyGradient = new List<Matrix>();
            List<Matrix> fullyBiasGradient = new List<Matrix>();
            List<Matrix> fully2Gradient = new List<Matrix>();
            List<Matrix> fully2BiasGradient = new List<Matrix>();
            List<Matrix> gGradient = new List<Matrix>();
            List<Matrix> csGradient = new List<Matrix>();
            List<Matrix> betaGradient = new List<Matrix>();
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
                fullyGradient.Add(this.outputLayers[i].GradientMatrix("FW"));
                fullyBiasGradient.Add(this.outputLayers[i].GradientMatrix("FB"));
                fully2Gradient.Add(this.outputLayers[i].GradientMatrix("F2W"));
                fully2BiasGradient.Add(this.outputLayers[i].GradientMatrix("F2B"));
                gGradient.Add(this.outputLayers[i].GradientMatrix("G"));
                csGradient.Add(this.outputLayers[i].GradientMatrix("CS"));
                betaGradient.Add(this.outputLayers[i].GradientMatrix("Beta"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new EmbeddingComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("Input", _ => this.Input)
                .AddScalar("Divisor", x => 1d / Math.Pow(this.NumNodes, 2))
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
                .AddWeight("FW", x => fully[x.Layer]).AddGradient("DFW", x => fullyGradient[x.Layer])
                .AddBias("FB", x => fullyBias[x.Layer]).AddGradient("DFB", x => fullyBiasGradient[x.Layer])
                .AddWeight("F2W", x => fully2[x.Layer]).AddGradient("DF2W", x => fully2Gradient[x.Layer])
                .AddBias("F2B", x => fully2Bias[x.Layer]).AddGradient("DF2B", x => fully2BiasGradient[x.Layer])
                .AddBias("G", x => g[x.Layer]).AddGradient("DG", x => gGradient[x.Layer])
                .AddBias("CS", x => cs[x.Layer]).AddGradient("DCS", x => csGradient[x.Layer])
                .AddBias("Beta", x => beta[x.Layer]).AddGradient("DBeta", x => betaGradient[x.Layer])
                .AddOperationFinder("nodeFeatures", x => x.Layer == 0 ? this.computationGraph["nodeFeatures_concatenate_0_0"] : this.computationGraph[$"output_act_0_{x.Layer - 1}"])
                .AddOperationFinder("output_act_array", x => this.computationGraph.ToOperationArray("output_act", new LayerInfo(0, 0), new LayerInfo(0, this.NumLayers - 1)))
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_softmax_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
