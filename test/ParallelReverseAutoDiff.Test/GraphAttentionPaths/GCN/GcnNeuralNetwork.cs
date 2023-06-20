using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN
{
    /// <summary>
    /// A GCN neural network.
    /// </summary>
    public partial class GcnNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN.Architecture";
        private const string ARCHITECTURE = "MessagePassing";

        private GcnComputationGraph computationGraph;

        private readonly List<IModelLayer> hiddenLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="GcnNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public GcnNeuralNetwork(int numLayers, int numPaths, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumPaths = numPaths;
            this.NumFeatures = numFeatures;

            this.hiddenLayers = new List<IModelLayer>();
            int numInputFeatures = numFeatures;
            int numInputOutputFeatures = numFeatures * 2;
            for (int i = 0; i < numLayers; ++i)
            {
                var hiddenLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("W", new[] { numInputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("B", new[] { 1, numInputOutputFeatures }, InitializationType.Zeroes);
                var hiddenLayer = hiddenLayerBuilder.Build();
                this.hiddenLayers.Add(hiddenLayer);
                numInputFeatures = numInputOutputFeatures;
                numInputOutputFeatures = numInputFeatures * 2;
            }

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input deep matrix.
        /// </summary>
        public DeepMatrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public Matrix[] Output { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets the adjacency matrix.
        /// </summary>
        public Matrix Adjacency { get; set; }

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
                .AddIntermediate("Output", x => this.Output[x.Layer])
                .AddWeight("W", x => w[x.Layer]).AddGradient("DW", x => wGrad[x.Layer])
                .AddBias("B", x => b[x.Layer]).AddGradient("DB", x => bGrad[x.Layer])
                .AddOperationFinder("Adjacency", _ => Adjacency)
                .AddOperationFinder("CurrentH", x => x.Layer == 0 ? this.computationGraph["input_trans_0_0"] : this.computationGraph[$"ah_w_act_0_{x.Layer - 1}"])
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph[$"ah_w_act_0_{this.NumLayers - 1}"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        public void AutomaticForwardPropagate(DeepMatrix input, bool doNotUpdate)
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
            var loss = new Matrix(1, 1); // meanSquaredErrorLossOperation.Forward(this.Output, this.Target);
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
            var gradientOfLossWrtOutput = new Matrix(1, 1); // (lossFunction.Backward(this.Output).Item1 as Matrix) ?? throw new Exception("Gradient of the loss wrt the output should not be null.");
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
            clearer.Clear(this.hiddenLayers.ToArray());

            // Clear intermediates
            this.Output = new Matrix[NumLayers];
            int numFeatures = this.NumFeatures;
            for (int i = 0; i < NumLayers; i++)
            {
                numFeatures *= 2;
                this.Output[i] = CommonMatrixUtils.InitializeZeroMatrix(this.NumPaths, numFeatures);
            }
            this.Parameters.InputSequence = CommonMatrixUtils.InitializeZeroMatrix(this.NumLayers, this.NumPaths, this.NumFeatures);
        }
    }
}
