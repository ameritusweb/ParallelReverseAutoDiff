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

        /// <summary>
        /// Initializes a new instance of the <see cref="GcnNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public GcnNeuralNetwork(int numLayers, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
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
            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new GcnComputationGraph(this);
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
