using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN
{
    /// <summary>
    /// An LSTM neural network.
    /// </summary>
    public class LstmNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.LSTM.Architecture";
        private const string ARCHITECTURE = "NodeProcessing";

        private LstmComputationGraph computationGraph;

        private readonly int hiddenSize;
        private readonly int originalInputSize;
        private readonly int inputSize;
        private readonly int outputSize;
        private readonly int numTimeSteps;

        private readonly IModelLayer embeddingLayer;
        private readonly IModelLayer hiddenLayer;
        private readonly IModelLayer outputLayer;

        private Matrix[][] h;
        private Matrix[][] c; // Memory cell state

        private Matrix[][][] arrays4D;
        private Matrix[][] arrays3D;


        /// <summary>
        /// Initializes a new instance of the <see cref="LstmNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public LstmNeuralNetwork(int inputSize, int hiddenSize, int outputSize, int numTimeSteps, int numLayers, double learningRate, double clipValue)
        {
            this.inputSize = hiddenSize;
            this.originalInputSize = inputSize;
            inputSize = hiddenSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.Parameters.NumTimeSteps = numTimeSteps;
            this.Parameters.LearningRate = learningRate;
            this.numTimeSteps = numTimeSteps;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;

            var embeddingLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("We", new[] { hiddenSize, this.originalInputSize }, InitializationType.Xavier)
                .AddModelElementGroup("be", new[] { hiddenSize, outputSize }, InitializationType.Zeroes);
            this.embeddingLayer = embeddingLayerBuilder.Build();

            var hiddenLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Wo", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Uo", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
                .AddModelElementGroup("bo", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wi", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Ui", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
                .AddModelElementGroup("bi", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wf", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Uf", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
                .AddModelElementGroup("bf", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wc", new[] { numLayers, hiddenSize, inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Uc", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
                .AddModelElementGroup("bc", new[] { numLayers, hiddenSize, outputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wq", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
                .AddModelElementGroup("Wk", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier)
                .AddModelElementGroup("Wv", new[] { numLayers, hiddenSize, hiddenSize }, InitializationType.Xavier);
            this.hiddenLayer = hiddenLayerBuilder.Build();

            var outputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("V", new[] { outputSize, hiddenSize }, InitializationType.Xavier)
                .AddModelElementGroup("b", new[] { outputSize, 1 }, InitializationType.Zeroes);
            this.outputLayer = outputLayerBuilder.Build();

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input matrix.
        /// </summary>
        public DeepMatrix Input { get; private set; }

        /// <summary>
        /// Gets the output path features matrix.
        /// </summary>
        public Matrix[] OutputPathFeatures { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets the number of layers of the neural network.
        /// </summary>
        internal int NumLayers { get; private set; }

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
            var we = this.embeddingLayer.WeightMatrix("We");
            var be = this.embeddingLayer.WeightMatrix("be");

            var dwe = this.embeddingLayer.GradientMatrix("We");
            var dbe = this.embeddingLayer.GradientMatrix("be");

            var wf = this.hiddenLayer.WeightDeepMatrix("Wf");
            var wi = this.hiddenLayer.WeightDeepMatrix("Wi");
            var wc = this.hiddenLayer.WeightDeepMatrix("Wc");
            var wo = this.hiddenLayer.WeightDeepMatrix("Wo");

            var dwf = this.hiddenLayer.GradientDeepMatrix("Wf");
            var dwi = this.hiddenLayer.GradientDeepMatrix("Wi");
            var dwc = this.hiddenLayer.GradientDeepMatrix("Wc");
            var dwo = this.hiddenLayer.GradientDeepMatrix("Wo");

            var uf = this.hiddenLayer.WeightDeepMatrix("Uf");
            var ui = this.hiddenLayer.WeightDeepMatrix("Ui");
            var uc = this.hiddenLayer.WeightDeepMatrix("Uc");
            var uo = this.hiddenLayer.WeightDeepMatrix("Uo");

            var duf = this.hiddenLayer.GradientDeepMatrix("Uf");
            var dui = this.hiddenLayer.GradientDeepMatrix("Ui");
            var duc = this.hiddenLayer.GradientDeepMatrix("Uc");
            var duo = this.hiddenLayer.GradientDeepMatrix("Uo");

            var bf = this.hiddenLayer.WeightDeepMatrix("bf");
            var bi = this.hiddenLayer.WeightDeepMatrix("bi");
            var bc = this.hiddenLayer.WeightDeepMatrix("bc");
            var bo = this.hiddenLayer.WeightDeepMatrix("bo");

            var dbf = this.hiddenLayer.GradientDeepMatrix("bf");
            var dbi = this.hiddenLayer.GradientDeepMatrix("bi");
            var dbc = this.hiddenLayer.GradientDeepMatrix("bc");
            var dbo = this.hiddenLayer.GradientDeepMatrix("bo");

            var wq = this.hiddenLayer.WeightDeepMatrix("Wq");
            var wk = this.hiddenLayer.WeightDeepMatrix("Wk");
            var wv = this.hiddenLayer.WeightDeepMatrix("Wv");

            var dwq = this.hiddenLayer.GradientDeepMatrix("Wf");
            var dwk = this.hiddenLayer.GradientDeepMatrix("Wi");
            var dwv = this.hiddenLayer.GradientDeepMatrix("Wc");

            var v = this.outputLayer.WeightMatrix("V");
            var b = this.outputLayer.WeightMatrix("b");

            var dv = this.outputLayer.GradientMatrix("V");
            var db = this.outputLayer.GradientMatrix("b");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new LstmComputationGraph(this);
            var zeroMatrixHiddenSize = new Matrix(this.hiddenSize, 1);
            this.computationGraph
                .AddIntermediate("InputNodeFeatures", x => this.Parameters.InputSequence[x.TimeStep])
                .AddIntermediate("OutputPathFeatures", x => this.OutputPathFeatures[x.TimeStep])
                .AddIntermediate("c", x => this.c[x.TimeStep][x.Layer])
                .AddIntermediate("h", x => this.h[x.TimeStep][x.Layer])
                .AddScalar("scaledDotProductScalar", x => 1.0d / Math.Sqrt(this.hiddenSize))
                .AddWeight("Wf", x => wf[x.Layer]).AddGradient("dWf", x => dwf[x.Layer])
                .AddWeight("Wi", x => wi[x.Layer]).AddGradient("dWi", x => dwi[x.Layer])
                .AddWeight("Wc", x => wc[x.Layer]).AddGradient("dWc", x => dwc[x.Layer])
                .AddWeight("Wo", x => wo[x.Layer]).AddGradient("dWo", x => dwo[x.Layer])
                .AddWeight("Uf", x => uf[x.Layer]).AddGradient("dUf", x => duf[x.Layer])
                .AddWeight("Ui", x => ui[x.Layer]).AddGradient("dUi", x => dui[x.Layer])
                .AddWeight("Uc", x => uc[x.Layer]).AddGradient("dUc", x => duc[x.Layer])
                .AddWeight("Uo", x => uo[x.Layer]).AddGradient("dUo", x => duo[x.Layer])
                .AddWeight("bf", x => bf[x.Layer]).AddGradient("dbf", x => dbf[x.Layer])
                .AddWeight("bi", x => bi[x.Layer]).AddGradient("dbi", x => dbi[x.Layer])
                .AddWeight("bc", x => bc[x.Layer]).AddGradient("dbc", x => dbc[x.Layer])
                .AddWeight("bo", x => bo[x.Layer]).AddGradient("dbo", x => dbo[x.Layer])
                .AddWeight("Wq", x => wq[x.Layer]).AddGradient("dWq", x => dwq[x.Layer])
                .AddWeight("Wk", x => wk[x.Layer]).AddGradient("dWk", x => dwk[x.Layer])
                .AddWeight("Wv", x => wv[x.Layer]).AddGradient("dWv", x => dwv[x.Layer])
                .AddWeight("We", x => we).AddGradient("dWe", x => dwe)
                .AddWeight("be", x => be).AddGradient("dbe", x => dbe)
                .AddWeight("V", x => v).AddGradient("dV", x => dv)
                .AddWeight("b", x => b).AddGradient("db", x => db)
                .AddOperationFinder("i", x => this.computationGraph[$"i_{x.TimeStep}_{x.Layer}"])
                .AddOperationFinder("f", x => this.computationGraph[$"f_{x.TimeStep}_{x.Layer}"])
                .AddOperationFinder("cHat", x => this.computationGraph[$"cHat_{x.TimeStep}_{x.Layer}"])
                .AddOperationFinder("o", x => this.computationGraph[$"o_{x.TimeStep}_{x.Layer}"])
                .AddOperationFinder("embeddedInput", x => this.computationGraph[$"embeddedInput_{x.TimeStep}_0"])
                .AddOperationFinder("hFromCurrentTimeStepAndLastLayer", x => this.computationGraph[$"h_{x.TimeStep}_{this.NumLayers - 1}"])
                .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph[$"embeddedInput_{x.TimeStep}_0"] : this.computationGraph[$"h_{x.TimeStep}_{x.Layer - 1}"])
                .AddOperationFinder("previousHiddenState", x => x.TimeStep == 0 ? zeroMatrixHiddenSize : this.computationGraph[$"h_{x.TimeStep - 1}_{x.Layer}"])
                .AddOperationFinder("previousMemoryCellState", x => x.TimeStep == 0 ? zeroMatrixHiddenSize : this.computationGraph[$"c_{x.TimeStep - 1}_{x.Layer}"])
                .ConstructFromArchitecture(jsonArchitecture, this.numTimeSteps, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            for (int t = this.Parameters.NumTimeSteps - 1; t >= 0; t--)
            {
                backwardStartOperation = this.computationGraph[$"output_t_{t}_0"];
                OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                await opVisitor.TraverseAsync();
                await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
            }
        }

        public void StoreOperationIntermediates(Guid id)
        {
            this.computationGraph.StoreOperationIntermediates(id);
        }

        public void RestoreOperationIntermediates(Guid id)
        {
            this.computationGraph.RestoreOperationIntermediates(id);
        }

        public void AutomaticForwardPropagate(DeepMatrix input, bool doNotUpdate)
        {
            // Initialize hidden state, gradients, biases, and intermediates
            this.ClearState();

            CommonMatrixUtils.SetInPlace(this.Input.ToArray(), input.ToArray());
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

        public async Task<Matrix> AutomaticBackwardPropagate(Matrix gradient)
        {
            int traverseCount = 0;
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_t_0_0"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = true;
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
                traverseCount++;
            }
            IOperationBase? backwardEndOperation = this.computationGraph["InputNodeFeatures_0_0"];
            return backwardEndOperation.CalculatedGradient[0] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        private void InitializeState()
        {
            // Clear the hidden state and memory cell state
            this.h = new Matrix[this.Parameters.NumTimeSteps][];
            this.c = new Matrix[this.Parameters.NumTimeSteps][];
            for (int t = 0; t < this.Parameters.NumTimeSteps; ++t)
            {
                this.h[t] = CommonMatrixUtils.InitializeZeroMatrix(this.NumLayers, this.hiddenSize, 1);
                this.c[t] = CommonMatrixUtils.InitializeZeroMatrix(this.NumLayers, this.hiddenSize, 1);
            }

            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });

            // Clear intermediates
            this.OutputPathFeatures = CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.outputSize, 1);
            this.Parameters.InputSequence = CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.originalInputSize, 1);

            this.arrays4D = new Matrix[][][] { this.h, this.c };
            this.arrays3D = new Matrix[][] { this.OutputPathFeatures, this.Parameters.InputSequence };
        }
    }
}
