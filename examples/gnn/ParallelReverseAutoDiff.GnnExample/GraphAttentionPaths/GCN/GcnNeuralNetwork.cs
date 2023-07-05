// ------------------------------------------------------------------------------
// <copyright file="GcnNeuralNetwork.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A GCN neural network.
    /// </summary>
    public class GcnNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN.Architecture";
        private const string ARCHITECTURE = "MessagePassing";

        private readonly List<IModelLayer> hiddenLayers;

        private GcnComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="GcnNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numPaths">The number of paths.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public GcnNeuralNetwork(int numLayers, int numPaths, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumPaths = numPaths;
            this.NumFeatures = numFeatures * (int)Math.Pow(2d, numLayers) * 2;

            this.hiddenLayers = new List<IModelLayer>();
            int numInputFeatures = this.NumFeatures;
            int numInputOutputFeatures = this.NumFeatures;
            for (int i = 0; i < numLayers; ++i)
            {
                var hiddenLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("W", new[] { numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("B", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes);
                var hiddenLayer = hiddenLayerBuilder.Build();
                this.hiddenLayers.Add(hiddenLayer);
                numInputFeatures = numInputOutputFeatures;
                numInputOutputFeatures = numInputFeatures;
            }

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input deep matrix.
        /// </summary>
        public FourDimensionalMatrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public FourDimensionalMatrix Output { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets or sets the adjacency matrix.
        /// </summary>
        public DeepMatrix Adjacency { get; set; }

        /// <summary>
        /// Gets the model layers for the GCN neural network.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return this.hiddenLayers;
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
        /// The forward pass of the GCN neural network.
        /// </summary>
        /// <param name="input">The input.</param>
        public void AutomaticForwardPropagate(FourDimensionalMatrix input)
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
                if (op.ResultToName != null)
                {
                    var split = op.ResultToName.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                    var oo = this.computationGraph[MatrixType.Intermediate, split[0], op.LayerInfo];
                    op.ReplaceResult(oo);
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
        /// The backward pass of the GCN neural network.
        /// </summary>
        /// <param name="gradient">The gradient of the loss.</param>
        /// <returns>The gradient.</returns>
        public async Task<DeepMatrix[]> AutomaticBackwardPropagate(DeepMatrix gradient)
        {
            IOperationBase? backwardStartOperation = this.computationGraph[$"ah_w_act_0_{this.NumLayers - 1}"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = true;
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
            }

            IOperationBase? backwardEndOperation = this.computationGraph["input_0_0"];
            var matrixArray = backwardEndOperation.CalculatedGradient.OfType<DeepMatrix>().ToArray();
            return matrixArray;
        }

        /// <summary>
        /// Initializes the state of the GCN neural network.
        /// </summary>
        public void InitializeState()
        {
            // Clear intermediates
            var output = new DeepMatrix[this.NumLayers];
            int numFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; i++)
            {
                numFeatures *= 2;
                output[i] = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumPaths, numFeatures));
            }

            var adjacency = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumPaths, this.NumPaths));
            var input = CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumPaths, this.NumFeatures, 1).Select(x => new DeepMatrix(x)).ToArray();

            if (this.Output == null)
            {
                this.Output = new FourDimensionalMatrix(output);
            }
            else
            {
                CommonMatrixUtils.SetInPlaceReplace(this.Output, new FourDimensionalMatrix(output));
            }

            if (this.Adjacency == null)
            {
                this.Adjacency = adjacency;
            }
            else
            {
                this.Adjacency.Replace(adjacency.ToArray());
            }

            if (this.Input == null)
            {
                this.Input = new FourDimensionalMatrix(input);
            }
            else
            {
                this.Input.Replace(input);
            }
        }

        /// <summary>
        /// Clears the state of the GCN neural network.
        /// </summary>
        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.hiddenLayers.ToArray());
        }

        /// <summary>
        /// Initializes the computation graph of the GCN neural network.
        /// </summary>
        /// <returns>A task.</returns>
        private async Task InitializeComputationGraph()
        {
            List<Matrix> w = new List<Matrix>();
            List<Matrix> b = new List<Matrix>();
            List<Matrix> wGrad = new List<Matrix>();
            List<Matrix> bGrad = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                w.Add(this.hiddenLayers[i].WeightMatrix("W"));
                b.Add(this.hiddenLayers[i].WeightMatrix("B"));
                wGrad.Add(this.hiddenLayers[i].GradientMatrix("W"));
                bGrad.Add(this.hiddenLayers[i].GradientMatrix("B"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new GcnComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Input", _ => this.Input)
                .AddIntermediate("Output", x => this.Output[x.Layer])
                .AddWeight("W", x => w[x.Layer]).AddGradient("DW", x => wGrad[x.Layer])
                .AddBias("B", x => b[x.Layer]).AddGradient("DB", x => bGrad[x.Layer])
                .AddOperationFinder("Adjacency", _ => this.Adjacency)
                .AddOperationFinder("CurrentH", x => x.Layer == 0 ? this.computationGraph["input_trans_0_0"] : this.computationGraph[$"ah_w_act_0_{x.Layer - 1}"])
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph[$"ah_w_act_0_{this.NumLayers - 1}"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
