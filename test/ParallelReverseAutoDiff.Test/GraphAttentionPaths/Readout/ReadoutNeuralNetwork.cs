using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using ParallelReverseAutoDiff.Test.FeedForward.RMAD;

namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.GCN
{
    /// <summary>
    /// A readout neural network.
    /// </summary>
    public partial class ReadoutNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.Test.GraphAttentionPaths.Readout.Architecture";
        private const string ARCHITECTURE = "Readout";

        private ReadoutComputationGraph computationGraph;

        private readonly List<IModelLayer> inputLayers;
        private readonly List<IModelLayer> nestedLayers;
        private readonly List<IModelLayer> outputLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReadoutNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numQueries">The number of queries.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public ReadoutNeuralNetwork(int numLayers, int numQueries, int numPaths, int numFeatures, double learningRate, double clipValue)
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
                outputFeaturesList.Add(numNestedOutputFeatures);
            }

            this.outputLayers = new List<IModelLayer>();
            for (int i = 0; i < numLayers; ++i)
            {
                var outputLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("R", new[] { outputFeaturesList[i], numFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("RB", new[] { 1, numFeatures }, InitializationType.Zeroes);
                var outputLayer = outputLayerBuilder.Build();
                this.outputLayers.Add(outputLayer);
            }
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
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

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
        /// Optimize the neural network.
        /// </summary>
        /// <param name="input">The input matrix.</param>
        /// <param name="target">The target matrix.</param>
        /// <param name="iterationIndex">The iteration index.</param>
        /// <param name="doNotUpdate">Whether or not the parameters should be updated.</param>
        /// <returns>A task.</returns>
        public async Task Optimize(DeepMatrix input, Matrix target, int iterationIndex, bool? doNotUpdate)
        {
            this.Target = target;
            if (doNotUpdate == null)
            {
                doNotUpdate = false;
            }

            this.Parameters.AdamIteration = iterationIndex + 1;

            await this.AutomaticForwardPropagate(input, doNotUpdate.Value);
        }

        private void ClearState()
        {

        }

        private async Task InitializeComputationGraph()
        {
            List<Matrix> keys = new List<Matrix>();
            List<Matrix> keysBias = new List<Matrix>();
            List<Matrix> values = new List<Matrix>();
            List<Matrix> valuesBias = new List<Matrix>();
            List<Matrix> queries = new List<Matrix>();
            List<Matrix> queriesBias = new List<Matrix>();
            List<Matrix> reduce = new List<Matrix>();
            List<Matrix> reduceBias = new List<Matrix>();
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
            }

            List<Matrix> keysGradient = new List<Matrix>();
            List<Matrix> keysBiasGradient = new List<Matrix>();
            List<Matrix> valuesGradient = new List<Matrix>();
            List<Matrix> valuesBiasGradient = new List<Matrix>();
            List<Matrix> queriesGradient = new List<Matrix>();
            List<Matrix> queriesBiasGradient = new List<Matrix>();
            List<Matrix> reduceGradient = new List<Matrix>();
            List<Matrix> reduceBiasGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                keys.Add(this.inputLayers[i].GradientMatrix("Keys"));
                keysBias.Add(this.inputLayers[i].GradientMatrix("KB"));
                values.Add(this.inputLayers[i].GradientMatrix("Values"));
                valuesBias.Add(this.inputLayers[i].GradientMatrix("VB"));
                queries.Add(this.nestedLayers[i].GradientMatrix("Queries"));
                queriesBias.Add(this.nestedLayers[i].GradientMatrix("QB"));
                reduce.Add(this.outputLayers[i].GradientMatrix("R"));
                reduceBias.Add(this.outputLayers[i].GradientMatrix("RB"));
            }

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<NestedLayersJsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new ReadoutComputationGraph(this);
            //this.computationGraph
            //    .AddIntermediate("Output", _ => this.Output)
            //    .AddIntermediate("H", x => this.HiddenLayers[x.Layer].H)
            //    .AddWeight("Cf1", x => this.FirstConvolutionalLayers[x.Layer].Cf1).AddGradient("DCf1", x => this.FirstConvolutionalLayers[x.Layer].DCf1)
            //    .AddBias("Cb1", x => this.FirstConvolutionalLayers[x.Layer].Cb1).AddGradient("DCb1", x => this.FirstConvolutionalLayers[x.Layer].DCb1)
            //    .AddWeight("Sc1", x => this.FirstConvolutionalLayers[x.Layer].Sc1).AddGradient("DSc1", x => this.FirstConvolutionalLayers[x.Layer].DSc1)
            //    .AddWeight("Sh1", x => this.FirstConvolutionalLayers[x.Layer].Sh1).AddGradient("DSh1", x => this.FirstConvolutionalLayers[x.Layer].DSh1)
            //    .AddWeight("Cf2", x => this.SecondConvolutionalLayers[x.Layer].Cf2).AddGradient("DCf2", x => this.SecondConvolutionalLayers[x.Layer].DCf2)
            //    .AddBias("Cb2", x => this.SecondConvolutionalLayers[x.Layer].Cb2).AddGradient("DCb2", x => this.SecondConvolutionalLayers[x.Layer].DCb2)
            //    .AddWeight("Sc2", x => this.SecondConvolutionalLayers[x.Layer].Sc2).AddGradient("DSc2", x => this.SecondConvolutionalLayers[x.Layer].DSc2)
            //    .AddWeight("Sh2", x => this.SecondConvolutionalLayers[x.Layer].Sh2).AddGradient("DSh2", x => this.SecondConvolutionalLayers[x.Layer].DSh2)
            //    .AddWeight("We", _ => this.EmbeddingLayer.We).AddGradient("DWe", _ => this.EmbeddingLayer.DWe)
            //    .AddBias("Be", _ => this.EmbeddingLayer.Be).AddGradient("DBe", _ => this.EmbeddingLayer.DBe)
            //    .AddWeight("W", x => this.HiddenLayers[x.Layer].W).AddGradient("DW", x => this.HiddenLayers[x.Layer].DW)
            //    .AddBias("B", x => this.HiddenLayers[x.Layer].B).AddGradient("DB", x => this.HiddenLayers[x.Layer].DB)
            //    .AddWeight("V", _ => this.OutputLayer.V).AddGradient("DV", _ => this.OutputLayer.DV)
            //    .AddBias("Bo", _ => this.OutputLayer.Bo).AddGradient("DBo", _ => this.OutputLayer.DBo)
            //    .AddWeight("ScEnd2", x => this.HiddenLayers[x.Layer].ScEnd2).AddGradient("DScEnd2", x => this.HiddenLayers[x.Layer].DScEnd2)
            //    .AddWeight("ShEnd2", x => this.HiddenLayers[x.Layer].ShEnd2).AddGradient("DShEnd2", x => this.HiddenLayers[x.Layer].DShEnd2)
            //    .AddOperationFinder("ActivatedFromLastLayer1", _ => this.computationGraph[$"activated1_0_{this.NumLayers - 1}"])
            //    .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.Input : this.computationGraph[$"activated1_0_{x.Layer - 1}"])
            //    .AddOperationFinder("currentInputOrMaxPooling", x => x.Layer == 0 ? this.computationGraph[$"maxPooling1_0_0"] : this.computationGraph[$"activated2_0_{x.Layer - 1}"])
            //    .AddOperationFinder("ActivatedFromLastLayer2", _ => this.computationGraph[$"activated2_0_{this.NumLayers - 1}"])
            //    .AddOperationFinder("thirdLayerCurrentInput", x => x.Layer == 0 ? this.computationGraph["embeddedInput_0_0"] : this.computationGraph[$"h_act_0_{x.Layer - 1}"])
            //    .AddOperationFinder("HFromLastLayer", _ => this.computationGraph[$"h_act_0_{this.NumLayers - 1}"])
            //    .ConstructFromArchitecture(jsonArchitecture, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_t_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        private async Task AutomaticForwardPropagate(DeepMatrix input, bool doNotUpdate)
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
    }
}
