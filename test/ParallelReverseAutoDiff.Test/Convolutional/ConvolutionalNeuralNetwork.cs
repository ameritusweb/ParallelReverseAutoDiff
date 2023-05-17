using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.Convolutional
{
    /// <summary>
    /// A convolutional neural network.
    /// </summary>
    public partial class ConvolutionalNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.Convolutional.Architecture";
        private const string ARCHITECTURE = "ConvolutionalArchitecture";
        private EmbeddingLayer embeddingLayer;
        private OutputLayer outputLayer;

        private ConvolutionalComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="ConvolutionalNeuralNetwork"/> class.
        /// </summary>
        /// <param name="inputSize">The input size.</param>
        /// <param name="hiddenSize">The hidden size.</param>
        /// <param name="outputSize">The output size.</param>
        /// <param name="numFilters">The number of filters.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public ConvolutionalNeuralNetwork(Dimension inputDimensions, Dimension filterDimensions, int inputSize, int hiddenSize, int outputSize, int numFilters, int numLayers, double learningRate, double? clipValue)
        {
            this.InputDimensions = inputDimensions;
            this.FilterDimensions = filterDimensions;
            this.InputSize = hiddenSize;
            this.OriginalInputSize = inputSize;
            this.HiddenSize = hiddenSize;
            this.OutputSize = outputSize;
            this.Parameters.LearningRate = learningRate;
            this.NumFilters = numFilters;
            this.NumLayers = numLayers;
            if (clipValue != null)
            {
                this.ClipValue = clipValue.Value;
            }

            this.Input = new DeepMatrix(inputDimensions);

            this.FirstConvolutionalLayers = new FirstConvolutionalLayer[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                this.FirstConvolutionalLayers[i] = new FirstConvolutionalLayer(this);
                var firstConvolutionalLayer = this.FirstConvolutionalLayers[i];
                firstConvolutionalLayer.Initialize();
                firstConvolutionalLayer.InitializeGradients();
            }

            this.SecondConvolutionalLayers = new SecondConvolutionalLayer[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                this.SecondConvolutionalLayers[i] = new SecondConvolutionalLayer(this);
                var secondConvolutionalLayer = this.SecondConvolutionalLayers[i];
                secondConvolutionalLayer.Initialize();
                secondConvolutionalLayer.InitializeGradients();
            }

            this.EmbeddingLayer.Initialize();
            this.EmbeddingLayer.InitializeGradients();

            this.HiddenLayers = new HiddenLayer[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                this.HiddenLayers[i] = new HiddenLayer(this);
                var hiddenLayer = this.HiddenLayers[i];
                hiddenLayer.Initialize();
                hiddenLayer.InitializeGradients();
            }

            this.Output = new Matrix(outputSize, 1);

            this.OutputLayer.Initialize();
            this.OutputLayer.InitializeGradients();
        }

        /// <summary>
        /// Gets the input matrix.
        /// </summary>
        public DeepMatrix Input { get; private set; }

        /// <summary>
        /// Gets or sets the first convolutional layers.
        /// </summary>
        public FirstConvolutionalLayer[] FirstConvolutionalLayers { get; set; }

        /// <summary>
        /// Gets or sets the second convolutional layers.
        /// </summary>
        public SecondConvolutionalLayer[] SecondConvolutionalLayers { get; set; }

        /// <summary>
        /// Gets the embedding layer.
        /// </summary>
        public EmbeddingLayer EmbeddingLayer
        {
            get
            {
                return this.embeddingLayer ??= new EmbeddingLayer(this);
            }
        }

        /// <summary>
        /// Gets or sets the hidden layers.
        /// </summary>
        public HiddenLayer[] HiddenLayers { get; set; }

        /// <summary>
        /// Gets the output layer.
        /// </summary>
        public OutputLayer OutputLayer
        {
            get
            {
                return this.outputLayer ??= new OutputLayer(this);
            }
        }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public Matrix Output { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets the input dimensions of the neural network.
        /// </summary>
        internal Dimension InputDimensions { get; private set; }

        /// <summary>
        /// Gets the filter dimensions of the neural network.
        /// </summary>
        internal Dimension FilterDimensions { get; private set; }

        /// <summary>
        /// Gets the input size of the neural network.
        /// </summary>
        internal int InputSize { get; private set; }

        /// <summary>
        /// Gets the original input size of the neural network.
        /// </summary>
        internal int OriginalInputSize { get; private set; }

        /// <summary>
        /// Gets the hidden size of the neural network.
        /// </summary>
        internal int HiddenSize { get; private set; }

        /// <summary>
        /// Gets the output size of the neural network.
        /// </summary>
        internal int OutputSize { get; private set; }

        /// <summary>
        /// Gets the number of filters of the neural network.
        /// </summary>
        internal int NumFilters { get; private set; }

        /// <summary>
        /// Gets the number of layers of the neural network.
        /// </summary>
        internal int NumLayers { get; private set; }

        /// <summary>
        /// Gets the clip value for the neural network.
        /// </summary>
        internal double ClipValue { get; private set; } = 4d;

        /// <summary>
        /// Gets the Adam iteration for the neural network.
        /// </summary>
        internal double AdamIteration { get; private set; }

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

            this.AdamIteration = iterationIndex + 1;

            await this.AutomaticForwardPropagate(input, doNotUpdate.Value);
        }

        private void ClearState()
        {
            Parallel.For(0, this.HiddenLayers.Length, (i) =>
            {
                this.HiddenLayers[i].ClearState();
            });
            this.OutputLayer.ClearState();
        }

        private async Task InitializeComputationGraph()
        {
            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<TripleLayersJsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new ConvolutionalComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("H", x => this.HiddenLayers[x.Layer].H)
                .AddWeight("We", _ => this.EmbeddingLayer.We)
                .AddBias("Be", _ => this.EmbeddingLayer.Be)
                .AddWeight("W", x => this.HiddenLayers[x.Layer].W)
                .AddBias("B", x => this.HiddenLayers[x.Layer].B)
                .AddWeight("V", _ => this.OutputLayer.V)
                .AddBias("Bo", _ => this.OutputLayer.Bo)
                .AddGradient("DWe", _ => this.EmbeddingLayer.DWe)
                .AddGradient("DBe", _ => this.EmbeddingLayer.DBe)
                .AddGradient("DW", x => this.HiddenLayers[x.Layer].DW)
                .AddGradient("DB", x => this.HiddenLayers[x.Layer].DB)
                .AddGradient("DV", _ => this.OutputLayer.DV)
                .AddGradient("DBo", _ => this.OutputLayer.DBo)
                .AddOperationFinder("HFromLastLayer", _ => this.computationGraph[$"h_act_0_{this.NumLayers - 1}"])
                .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph["embeddedInput_0_0"] : this.computationGraph[$"h_act_0_{x.Layer - 1}"])
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

            CommonMatrixUtils.SetInPlace(new[] { this.Input }, new[] { input });
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
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
                traverseCount++;
            }

            if (traverseCount == 0 || doNotUpdate)
            {
                return;
            }

            // Clip gradients and biases to prevent exploding gradients
            this.EmbeddingLayer.ClipGradients();
            Parallel.For(0, this.HiddenLayers.Length, (i) =>
            {
                this.HiddenLayers[i].ClipGradients();
            });

            this.OutputLayer.ClipGradients();

            // Update model parameters using gradient descent
            this.UpdateEmbeddingLayerParametersWithAdam(this.EmbeddingLayer);
            Parallel.For(0, this.HiddenLayers.Length, (i) =>
            {
                this.UpdateHiddenLayerParametersWithAdam(this.HiddenLayers[i]);
            });
            this.UpdateOutputLayerParametersWithAdam(this.OutputLayer);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// <summary>
    /// The Adam optimization for a feed forward neural network.
    /// </summary>
    public partial class ConvolutionalNeuralNetwork
    {
        private readonly double beta1 = 0.9;
        private readonly double beta2 = 0.999;
        private readonly double epsilon = 1e-8;

        private void UpdateEmbeddingLayerParametersWithAdam(EmbeddingLayer embeddingLayer)
        {
            this.UpdateWeightWithAdam(embeddingLayer.We, embeddingLayer.MWe, embeddingLayer.VWe, embeddingLayer.DWe, this.beta1, this.beta2, this.epsilon);
            this.UpdateWeightWithAdam(embeddingLayer.Be, embeddingLayer.MBe, embeddingLayer.VBe, embeddingLayer.DBe, this.beta1, this.beta2, this.epsilon);
        }

        private void UpdateHiddenLayerParametersWithAdam(HiddenLayer hiddenLayer)
        {
            this.UpdateWeightWithAdam(hiddenLayer.W, hiddenLayer.MW, hiddenLayer.VW, hiddenLayer.DW, this.beta1, this.beta2, this.epsilon);
            this.UpdateWeightWithAdam(hiddenLayer.B, hiddenLayer.MB, hiddenLayer.VB, hiddenLayer.DB, this.beta1, this.beta2, this.epsilon);
        }

        private void UpdateOutputLayerParametersWithAdam(OutputLayer outputLayer)
        {
            this.UpdateWeightWithAdam(outputLayer.V, outputLayer.MV, outputLayer.VV, outputLayer.DV, this.beta1, this.beta2, this.epsilon);
            this.UpdateWeightWithAdam(outputLayer.Bo, outputLayer.MBo, outputLayer.VBo, outputLayer.DBo, this.beta1, this.beta2, this.epsilon);
        }

        private void UpdateWeightWithAdam(Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon)
        {
            // Update biased first moment estimate
            mW = CommonMatrixUtils.MatrixAdd(CommonMatrixUtils.ScalarMultiply(beta1, mW), CommonMatrixUtils.ScalarMultiply(1 - beta1, gradient));

            // Update biased second raw moment estimate
            vW = CommonMatrixUtils.MatrixAdd(CommonMatrixUtils.ScalarMultiply(beta2, vW), CommonMatrixUtils.ScalarMultiply(1 - beta2, CommonMatrixUtils.HadamardProduct(gradient, gradient)));

            // Compute bias-corrected first moment estimate
            Matrix mW_hat = CommonMatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, this.AdamIteration)), mW);

            // Compute bias-corrected second raw moment estimate
            Matrix vW_hat = CommonMatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, this.AdamIteration)), vW);

            // Update weights
            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    double weightReductionValue = this.Parameters.LearningRate * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);
                    w[i][j] -= weightReductionValue;
                }
            }
        }
    }
}
