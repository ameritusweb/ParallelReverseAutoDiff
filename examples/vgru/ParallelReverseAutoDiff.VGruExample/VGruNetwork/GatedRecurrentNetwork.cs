// ------------------------------------------------------------------------------
// <copyright file="GatedRecurrentNetwork.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VLstmExample.Common;

    /// <summary>
    /// A vector gated recurrent network.
    /// </summary>
    public class GatedRecurrentNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.VGruExample.VGruNetwork.Architecture";
        private const string ARCHITECTURE = "vgrunet";

        private readonly IModelLayer inputLayer;
        private readonly List<IModelLayer> nestedLayers;
        private readonly IModelLayer outputLayer;

        private GatedRecurrentComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="GatedRecurrentNetwork"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public GatedRecurrentNetwork(int numTimeSteps, int numLayers, int numNodes, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumNodes = numNodes;
            this.NumFeatures = numFeatures;
            this.Parameters.NumTimeSteps = numTimeSteps;

            int numInputOutputFeatures = this.NumFeatures;
            var inputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Weights", new[] { numNodes, numInputOutputFeatures / 2 }, InitializationType.Xavier)
                .AddModelElementGroup("Vectors", new[] { numNodes, numInputOutputFeatures }, InitializationType.Xavier);
            var inputLayer = inputLayerBuilder.Build();
            this.inputLayer = inputLayer;

            this.nestedLayers = new List<IModelLayer>();
            int numNestedOutputFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("WUpdateWeights", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("UUpdateWeights", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("WUpdateVectors", new[] { numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("UUpdateVectors", new[] { numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("ZKeys", new[] { numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("ZKB", new[] { 1, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("WResetWeights", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("UResetWeights", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("WResetVectors", new[] { numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("UResetVectors", new[] { numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("RKeys", new[] { numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("RKB", new[] { 1, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("IKeys", new[] { numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("IKB", new[] { 1, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("UCWeights", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("CHKeys", new[] { numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("CHKB", new[] { 1, numNestedOutputFeatures * 2 }, InitializationType.Xavier);
                var nestedLayer = nestedLayerBuilder.Build();
                this.nestedLayers.Add(nestedLayer);
            }

            var outputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("RowSumWeights", new[] { numNodes, numInputOutputFeatures }, InitializationType.Xavier);
            var outputLayer = outputLayerBuilder.Build();
            this.outputLayer = outputLayer;

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input matrices.
        /// </summary>
        public DeepMatrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrices.
        /// </summary>
        public DeepMatrix Output { get; private set; }

        /// <summary>
        /// Gets the model layers of the neural network.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return new IModelLayer[] { this.inputLayer }.Concat(this.nestedLayers).Append(this.outputLayer);
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

            IOperationBase? backwardEndOperation = this.computationGraph["weights_0_0"];
            if (backwardEndOperation.CalculatedGradient[0] == null)
            {
                return gradient;
            }

            return backwardEndOperation.CalculatedGradient[0] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initialize the state of the edge attention neural network.
        /// </summary>
        public void InitializeState()
        {
            // Clear intermediates
            var output = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, 1, this.NumFeatures));
            var input = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, 1, this.NumFeatures));

            if (this.Output == null)
            {
                this.Output = output;
            }
            else
            {
                CommonMatrixUtils.SetInPlaceReplace(this.Output, output);
            }

            if (this.Input == null)
            {
                this.Input = input;
            }
            else
            {
                CommonMatrixUtils.SetInPlaceReplace(this.Input, input);
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
            var weights = this.inputLayer.WeightMatrix("Weights");
            var weightsGradient = this.inputLayer.GradientMatrix("Weights");
            var vectors = this.inputLayer.WeightMatrix("Vectors");
            var vectorsGradient = this.inputLayer.GradientMatrix("Vectors");

            List<Matrix> wUpdateWeights = new List<Matrix>();
            List<Matrix> wUpdateWeightsGradient = new List<Matrix>();
            List<Matrix> uUpdateWeights = new List<Matrix>();
            List<Matrix> uUpdateWeightsGradient = new List<Matrix>();
            List<Matrix> wUpdateVectors = new List<Matrix>();
            List<Matrix> wUpdateVectorsGradient = new List<Matrix>();
            List<Matrix> uUpdateVectors = new List<Matrix>();
            List<Matrix> uUpdateVectorsGradient = new List<Matrix>();
            List<Matrix> zKeys = new List<Matrix>();
            List<Matrix> zKeysGradient = new List<Matrix>();
            List<Matrix> zKB = new List<Matrix>();
            List<Matrix> zKBGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wUpdateWeights.Add(layer.WeightMatrix("WUpdateWeights"));
                wUpdateWeightsGradient.Add(layer.GradientMatrix("WUpdateWeights"));
                uUpdateWeights.Add(layer.WeightMatrix("UUpdateWeights"));
                uUpdateWeightsGradient.Add(layer.GradientMatrix("UUpdateWeights"));
                wUpdateVectors.Add(layer.WeightMatrix("WUpdateVectors"));
                wUpdateVectorsGradient.Add(layer.GradientMatrix("WUpdateVectors"));
                uUpdateVectors.Add(layer.WeightMatrix("UUpdateVectors"));
                uUpdateVectorsGradient.Add(layer.GradientMatrix("UUpdateVectors"));
                zKeys.Add(layer.WeightMatrix("ZKeys"));
                zKeysGradient.Add(layer.GradientMatrix("ZKeys"));
                zKB.Add(layer.WeightMatrix("ZKB"));
                zKBGradient.Add(layer.GradientMatrix("ZKB"));
            }

            List<Matrix> wResetWeights = new List<Matrix>();
            List<Matrix> wResetWeightsGradient = new List<Matrix>();
            List<Matrix> uResetWeights = new List<Matrix>();
            List<Matrix> uResetWeightsGradient = new List<Matrix>();
            List<Matrix> wResetVectors = new List<Matrix>();
            List<Matrix> wResetVectorsGradient = new List<Matrix>();
            List<Matrix> uResetVectors = new List<Matrix>();
            List<Matrix> uResetVectorsGradient = new List<Matrix>();
            List<Matrix> rKeys = new List<Matrix>();
            List<Matrix> rKeysGradient = new List<Matrix>();
            List<Matrix> rKB = new List<Matrix>();
            List<Matrix> rKBGradient = new List<Matrix>();
            List<Matrix> iKeys = new List<Matrix>();
            List<Matrix> iKeysGradient = new List<Matrix>();
            List<Matrix> iKB = new List<Matrix>();
            List<Matrix> iKBGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wResetWeights.Add(layer.WeightMatrix("WResetWeights"));
                wResetWeightsGradient.Add(layer.GradientMatrix("WResetWeights"));
                uResetWeights.Add(layer.WeightMatrix("UResetWeights"));
                uResetWeightsGradient.Add(layer.GradientMatrix("UResetWeights"));
                wResetVectors.Add(layer.WeightMatrix("WResetVectors"));
                wResetVectorsGradient.Add(layer.GradientMatrix("WResetVectors"));
                uResetVectors.Add(layer.WeightMatrix("UResetVectors"));
                uResetVectorsGradient.Add(layer.GradientMatrix("UResetVectors"));
                rKeys.Add(layer.WeightMatrix("RKeys"));
                rKeysGradient.Add(layer.GradientMatrix("RKeys"));
                rKB.Add(layer.WeightMatrix("RKB"));
                rKBGradient.Add(layer.GradientMatrix("RKB"));
                iKeys.Add(layer.WeightMatrix("IKeys"));
                iKeysGradient.Add(layer.GradientMatrix("IKeys"));
                iKB.Add(layer.WeightMatrix("IKB"));
                iKBGradient.Add(layer.GradientMatrix("IKB"));
            }

            List<Matrix> uCWeights = new List<Matrix>();
            List<Matrix> uCWeightsGradient = new List<Matrix>();
            List<Matrix> chKeys = new List<Matrix>();
            List<Matrix> chKeysGradient = new List<Matrix>();
            List<Matrix> chKB = new List<Matrix>();
            List<Matrix> chKBGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                uCWeights.Add(layer.WeightMatrix("UCWeights"));
                uCWeightsGradient.Add(layer.GradientMatrix("UCWeights"));
                chKeys.Add(layer.WeightMatrix("CHKeys"));
                chKeysGradient.Add(layer.GradientMatrix("CHKeys"));
                chKB.Add(layer.WeightMatrix("CHKB"));
                chKBGradient.Add(layer.GradientMatrix("CHKB"));
            }

            var rowSumWeights = this.outputLayer.WeightMatrix("RowSumWeights");
            var rowSumWeightsGradient = this.outputLayer.GradientMatrix("RowSumWeights");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new GatedRecurrentComputationGraph(this);
            var zeroMatrixHiddenState = new Matrix(this.NumNodes, this.NumFeatures * 2);
            this.computationGraph
                .AddIntermediate("Output", x => this.Output[x.TimeStep])
                .AddIntermediate("Input", x => this.Input[x.TimeStep])
                .AddWeight("Weights", x => weights).AddGradient("DWeights", x => weightsGradient)
                .AddWeight("Vectors", x => vectors).AddGradient("DVectors", x => vectorsGradient)
                .AddWeight("WUpdateWeights", x => wUpdateWeights[x.Layer]).AddGradient("DWUpdateWeights", x => wUpdateWeightsGradient[x.Layer])
                .AddWeight("UUpdateWeights", x => uUpdateWeights[x.Layer]).AddGradient("DUUpdateWeights", x => uUpdateWeightsGradient[x.Layer])
                .AddWeight("WUpdateVectors", x => wUpdateVectors[x.Layer]).AddGradient("DWUpdateVectors", x => wUpdateVectorsGradient[x.Layer])
                .AddWeight("UUpdateVectors", x => uUpdateVectors[x.Layer]).AddGradient("DUUpdateVectors", x => uUpdateVectorsGradient[x.Layer])
                .AddWeight("ZKeys", x => zKeys[x.Layer]).AddGradient("DZKeys", x => zKeysGradient[x.Layer])
                .AddWeight("ZKB", x => zKB[x.Layer]).AddGradient("DZKB", x => zKBGradient[x.Layer])
                .AddWeight("WResetWeights", x => wResetWeights[x.Layer]).AddGradient("DWResetWeights", x => wResetWeightsGradient[x.Layer])
                .AddWeight("UResetWeights", x => uResetWeights[x.Layer]).AddGradient("DUResetWeights", x => uResetWeightsGradient[x.Layer])
                .AddWeight("WResetVectors", x => wResetVectors[x.Layer]).AddGradient("DWResetVectors", x => wResetVectorsGradient[x.Layer])
                .AddWeight("UResetVectors", x => uResetVectors[x.Layer]).AddGradient("DUResetVectors", x => uResetVectorsGradient[x.Layer])
                .AddWeight("RKeys", x => rKeys[x.Layer]).AddGradient("DRKeys", x => rKeysGradient[x.Layer])
                .AddWeight("RKB", x => rKB[x.Layer]).AddGradient("DRKB", x => rKBGradient[x.Layer])
                .AddWeight("IKeys", x => iKeys[x.Layer]).AddGradient("DIKeys", x => iKeysGradient[x.Layer])
                .AddWeight("IKB", x => iKB[x.Layer]).AddGradient("DIKB", x => iKBGradient[x.Layer])
                .AddWeight("UCWeights", x => uCWeights[x.Layer]).AddGradient("DUCWeights", x => uCWeightsGradient[x.Layer])
                .AddWeight("CHKeys", x => chKeys[x.Layer]).AddGradient("DCHKeys", x => chKeysGradient[x.Layer])
                .AddWeight("CHKB", x => chKB[x.Layer]).AddGradient("DCHKB", x => chKBGradient[x.Layer])
                .AddWeight("RowSumWeights", x => rowSumWeights).AddGradient("DRowSumWeights", x => rowSumWeightsGradient)
                .AddOperationFinder("newHFromLastLayer", x => this.computationGraph[$"compute_new_hidden_state_{x.TimeStep}_{this.NumLayers - 1}"])
                .AddOperationFinder("newHFromFirstLayer", x => this.computationGraph[$"compute_new_hidden_state_{x.TimeStep}_0"])
                .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph[$"input_projection_{x.TimeStep}_0"] : this.computationGraph[$"compute_new_hidden_state_{x.TimeStep}_{x.Layer - 1}"])
                .AddOperationFinder("previousHiddenState", x => x.TimeStep == 0 ? zeroMatrixHiddenState : this.computationGraph[$"compute_new_hidden_state_{x.TimeStep - 1}_{x.Layer}"])
                .ConstructFromArchitecture(jsonArchitecture, this.Parameters.NumTimeSteps, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["output_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }

        private void AddCount(string identifier, int count)
        {
            IOperationBase backwardStartOperation2a = this.computationGraph[identifier];
            backwardStartOperation2a.BackwardDependencyCounts = new List<int>();
            backwardStartOperation2a.BackwardDependencyCounts.Add(count);
        }
    }
}
