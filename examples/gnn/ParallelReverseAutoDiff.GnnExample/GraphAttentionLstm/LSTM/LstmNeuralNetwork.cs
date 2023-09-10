// ------------------------------------------------------------------------------
// <copyright file="LstmNeuralNetwork.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionLstm.LSTM
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An LSTM neural network.
    /// </summary>
    public class LstmNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.GnnExample.GraphAttentionPaths.LSTM.Architecture";
        private const string ARCHITECTURE = "NodeProcessing";

        private readonly int hiddenSize;
        private readonly int inputSize;
        private readonly int outputSize;

        private readonly IModelLayer embeddingLayer;
        private readonly IModelLayer hiddenLayer;
        private readonly IModelLayer outputLayer;

        private LstmComputationGraph computationGraph;

        private Matrix zeroMatrixHiddenSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="LstmNeuralNetwork"/> class.
        /// </summary>
        /// <param name="inputSize">The input size.</param>
        /// <param name="hiddenSize">The hidden size.</param>
        /// <param name="outputSize">The output size.</param>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public LstmNeuralNetwork(int inputSize, int hiddenSize, int outputSize, int numTimeSteps, int numLayers, double learningRate, double clipValue)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.Parameters.NumTimeSteps = numTimeSteps;
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;

            var embeddingLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("We", new[] { this.hiddenSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("be", new[] { 1, this.inputSize }, InitializationType.Zeroes);
            this.embeddingLayer = embeddingLayerBuilder.Build();

            var hiddenLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Wo", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Uo", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("bo", new[] { this.NumLayers, this.hiddenSize, this.inputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wi", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Ui", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("bi", new[] { this.NumLayers, this.hiddenSize, this.inputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wf", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Uf", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("bf", new[] { this.NumLayers, this.hiddenSize, this.inputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wc", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Uc", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("bc", new[] { this.NumLayers, this.hiddenSize, this.inputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("Wq", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Wk", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Wv", new[] { this.NumLayers, this.inputSize, this.inputSize }, InitializationType.Xavier);
            this.hiddenLayer = hiddenLayerBuilder.Build();

            var outputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("V", new[] { this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("b", new[] { 1, this.inputSize }, InitializationType.Zeroes)
                .AddModelElementGroup("FW", new[] { this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("FB", new[] { 1, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("F2W", new[] { this.inputSize, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("F2B", new[] { 1, this.inputSize }, InitializationType.Xavier)
                .AddModelElementGroup("Beta", new[] { 1, 1 }, InitializationType.He);
            this.outputLayer = outputLayerBuilder.Build();

            this.InitializeState();
        }

        /// <summary>
        /// Gets the output path features matrix.
        /// </summary>
        public DeepMatrix OutputPathFeatures { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets the JSON architecture.
        /// </summary>
        public JsonArchitecture Architecture { get; private set; }

        /// <summary>
        /// Gets the model layers of the LSTM neural network.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer };
            }
        }

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

        /// <summary>
        /// Stores the intermediate values of the computation graph for the given operation.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void StoreOperationIntermediates(Guid id)
        {
            this.computationGraph.StoreOperationIntermediates(id);
        }

        /// <summary>
        /// Restores the intermediate values of the computation graph for the given operation.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void RestoreOperationIntermediates(Guid id)
        {
            this.computationGraph.RestoreOperationIntermediates(id);
        }

        /// <summary>
        /// The forward pass for the LSTM neural network.
        /// </summary>
        /// <param name="input">The input.</param>
        public void AutomaticForwardPropagate(DeepMatrix input)
        {
            // Initialize hidden state, gradients, biases, and intermediates
            this.ClearState();

            CommonMatrixUtils.SetInPlace(this.Parameters.InputSequence, input);
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
        /// The backward pass for the LSTM neural network.
        /// </summary>
        /// <param name="gradient">The gradient of the loss.</param>
        /// <returns>The gradient.</returns>
        public async Task<DeepMatrix> AutomaticBackwardPropagate(DeepMatrix gradient)
        {
            IOperationBase? backwardStartOperation = null;
            for (int t = this.Parameters.NumTimeSteps - 1; t >= 0; t--)
            {
                backwardStartOperation = this.computationGraph[$"output_t_{t}_0"];
                if (!CommonMatrixUtils.IsAllZeroes(gradient[t]))
                {
                    var backwardInput = gradient[t];
                    backwardStartOperation.BackwardInput = backwardInput;
                    OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                    await opVisitor.TraverseAsync();
                    opVisitor.Reset();
                }
            }

            DeepMatrix output = new DeepMatrix(this.Parameters.NumTimeSteps, this.hiddenSize, this.inputSize);
            for (int i = 0; i < this.Parameters.NumTimeSteps; ++i)
            {
                IOperationBase? backwardEndOperation = this.computationGraph[$"projectedInput_{i}_0"];
                output[i] = backwardEndOperation.CalculatedGradient[1] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
            }

            return output;
        }

        /// <summary>
        /// Initialize the state of the LSTM neural network.
        /// </summary>
        public void InitializeState()
        {
            // Clear intermediates
            var outputPathFeatures = CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.hiddenSize, this.inputSize);
            var deepInputSequence = CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.inputSize, this.inputSize);

            if (this.OutputPathFeatures == null)
            {
                this.OutputPathFeatures = new DeepMatrix(outputPathFeatures);
            }
            else
            {
                CommonMatrixUtils.SetInPlaceReplace(this.OutputPathFeatures, new DeepMatrix(outputPathFeatures));
            }

            if (this.Parameters.InputSequence == null)
            {
                this.Parameters.InputSequence = new DeepMatrix(deepInputSequence);
            }
            else
            {
                CommonMatrixUtils.SetInPlaceReplace(this.Parameters.InputSequence, new DeepMatrix(deepInputSequence));
            }

            if (this.zeroMatrixHiddenSize != null)
            {
                this.zeroMatrixHiddenSize.Replace(CommonMatrixUtils.InitializeZeroMatrix(this.hiddenSize, this.inputSize).ToArray());
            }
        }

        /// <summary>
        /// Clears the state of the LSTM neural network.
        /// </summary>
        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });
        }

        /// <summary>
        /// Initializes the computation graph of the LSTM neural network.
        /// </summary>
        /// <returns>A task.</returns>
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

            var fully = this.outputLayer.WeightMatrix("FW");
            var fullyBias = this.outputLayer.WeightMatrix("FB");
            var fully2 = this.outputLayer.WeightMatrix("F2W");
            var fully2Bias = this.outputLayer.WeightMatrix("F2B");
            var beta = this.outputLayer.WeightMatrix("Beta");

            var dfully = this.outputLayer.GradientMatrix("FW");
            var dfullyBias = this.outputLayer.GradientMatrix("FB");
            var dfully2 = this.outputLayer.GradientMatrix("F2W");
            var dfully2Bias = this.outputLayer.GradientMatrix("F2B");
            var dbeta = this.outputLayer.GradientMatrix("Beta");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = this.Architecture ?? JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.Architecture = jsonArchitecture;

            this.computationGraph = new LstmComputationGraph(this);
            this.zeroMatrixHiddenSize = new Matrix(this.hiddenSize, this.inputSize);
            this.computationGraph
                .AddIntermediate("InputNodeFeatures", x => this.Parameters.DeepInputSequence[x.TimeStep])
                .AddIntermediate("OutputPathFeatures", x => this.OutputPathFeatures[x.TimeStep])
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
                .AddWeight("FW", x => fully).AddGradient("DFW", x => dfully)
                .AddWeight("FB", x => fullyBias).AddGradient("DFB", x => dfullyBias)
                .AddWeight("F2W", x => fully2).AddGradient("DF2W", x => dfully2)
                .AddWeight("F2B", x => fully2Bias).AddGradient("DF2B", x => dfully2Bias)
                .AddWeight("Beta", x => beta).AddGradient("DBeta", x => dbeta)
                .AddOperationFinder("embeddedInput", x => this.computationGraph[$"embeddedInput_{x.TimeStep}_0"])
                .AddOperationFinder("hFromCurrentTimeStepAndLastLayer", x => this.computationGraph[$"h_{x.TimeStep}_{this.NumLayers - 1}"])
                .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph[$"embeddedInput_{x.TimeStep}_0"] : this.computationGraph[$"h_{x.TimeStep}_{x.Layer - 1}"])
                .AddOperationFinder("previousHiddenState", x => x.TimeStep == 0 ? this.zeroMatrixHiddenSize : this.computationGraph[$"h_{x.TimeStep - 1}_{x.Layer}"])
                .AddOperationFinder("previousMemoryCellState", x => x.TimeStep == 0 ? this.zeroMatrixHiddenSize : this.computationGraph[$"c_{x.TimeStep - 1}_{x.Layer}"])
                .AddOperationFinder("h_array", x => this.computationGraph.ToOperationArray("h", new LayerInfo(x.TimeStep, 0), new LayerInfo(x.TimeStep, this.NumLayers - 1)))
                .ConstructFromArchitecture(jsonArchitecture, this.Parameters.NumTimeSteps, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            for (int t = this.Parameters.NumTimeSteps - 1; t >= 0; t--)
            {
                backwardStartOperation = this.computationGraph[$"output_t_{t}_0"];
                OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                await opVisitor.TraverseAsync();
                await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
            }
        }
    }
}
