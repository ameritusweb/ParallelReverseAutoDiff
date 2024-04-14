// ------------------------------------------------------------------------------
// <copyright file="SpatialNetwork.cs" author="ameritusweb" date="12/18/2023">
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
    public class SpatialNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.VGruExample.VGruNetwork.Architecture";
        private const string ARCHITECTURE = "spatialnet";

        private readonly IModelLayer inputLayer;
        private readonly List<IModelLayer> nestedLayers;
        private readonly IModelLayer outputLayer;

        private SpatialComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="SpatialNetwork"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public SpatialNetwork(int numTimeSteps, int numLayers, int numNodes, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumNodes = numNodes;
            this.NumFeatures = numFeatures;
            this.Parameters.NumTimeSteps = numTimeSteps;

            int numInputOutputFeatures = this.NumFeatures;
            var inputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Weights", new[] { numNodes, numInputOutputFeatures / 10 }, InitializationType.Xavier)
                .AddModelElementGroup("Angles", new[] { numInputOutputFeatures / 10, numInputOutputFeatures / 10 }, InitializationType.Xavier)
                .AddModelElementGroup("Vectors", new[] { numNodes, numInputOutputFeatures }, InitializationType.Xavier);
            var inputLayer = inputLayerBuilder.Build();
            this.inputLayer = inputLayer;

            this.nestedLayers = new List<IModelLayer>();
            int numNestedOutputFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("WUpdateWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UUpdateWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("WUpdateVectors", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UUpdateVectors", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("ZKeys", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("ZKB", new[] { numTimeSteps, 1, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("WResetWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UResetWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("WResetVectors", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UResetVectors", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("RKeys", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("RKB", new[] { numTimeSteps, 1, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("IKeys", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("IKB", new[] { numTimeSteps, 1, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UCWeights", new[] { numTimeSteps, numNodes, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("CHKeys", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("CHKB", new[] { numTimeSteps, 1, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d);
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
            backwardStartOperation = this.computationGraph[$"output_{this.Parameters.NumTimeSteps - 1}_0"];

            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, this.Parameters.NumTimeSteps - 1);
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
            var output = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.NumNodes, this.NumNodes * 2));
            var input = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.NumNodes, this.NumFeatures));

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
            var angles = this.inputLayer.WeightMatrix("Angles");
            var anglesGradient = this.inputLayer.GradientMatrix("Angles");
            var vectors = this.inputLayer.WeightMatrix("Vectors");
            var vectorsGradient = this.inputLayer.GradientMatrix("Vectors");

            List<DeepMatrix> wUpdateWeights = new List<DeepMatrix>();
            List<DeepMatrix> wUpdateWeightsGradient = new List<DeepMatrix>();
            List<DeepMatrix> uUpdateWeights = new List<DeepMatrix>();
            List<DeepMatrix> uUpdateWeightsGradient = new List<DeepMatrix>();
            List<DeepMatrix> wUpdateVectors = new List<DeepMatrix>();
            List<DeepMatrix> wUpdateVectorsGradient = new List<DeepMatrix>();
            List<DeepMatrix> uUpdateVectors = new List<DeepMatrix>();
            List<DeepMatrix> uUpdateVectorsGradient = new List<DeepMatrix>();
            List<DeepMatrix> zKeys = new List<DeepMatrix>();
            List<DeepMatrix> zKeysGradient = new List<DeepMatrix>();
            List<DeepMatrix> zKB = new List<DeepMatrix>();
            List<DeepMatrix> zKBGradient = new List<DeepMatrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wUpdateWeights.Add(layer.WeightDeepMatrix("WUpdateWeights"));
                wUpdateWeightsGradient.Add(layer.GradientDeepMatrix("WUpdateWeights"));
                uUpdateWeights.Add(layer.WeightDeepMatrix("UUpdateWeights"));
                uUpdateWeightsGradient.Add(layer.GradientDeepMatrix("UUpdateWeights"));
                wUpdateVectors.Add(layer.WeightDeepMatrix("WUpdateVectors"));
                wUpdateVectorsGradient.Add(layer.GradientDeepMatrix("WUpdateVectors"));
                uUpdateVectors.Add(layer.WeightDeepMatrix("UUpdateVectors"));
                uUpdateVectorsGradient.Add(layer.GradientDeepMatrix("UUpdateVectors"));
                zKeys.Add(layer.WeightDeepMatrix("ZKeys"));
                zKeysGradient.Add(layer.GradientDeepMatrix("ZKeys"));
                zKB.Add(layer.WeightDeepMatrix("ZKB"));
                zKBGradient.Add(layer.GradientDeepMatrix("ZKB"));
            }

            List<DeepMatrix> wResetWeights = new List<DeepMatrix>();
            List<DeepMatrix> wResetWeightsGradient = new List<DeepMatrix>();
            List<DeepMatrix> uResetWeights = new List<DeepMatrix>();
            List<DeepMatrix> uResetWeightsGradient = new List<DeepMatrix>();
            List<DeepMatrix> wResetVectors = new List<DeepMatrix>();
            List<DeepMatrix> wResetVectorsGradient = new List<DeepMatrix>();
            List<DeepMatrix> uResetVectors = new List<DeepMatrix>();
            List<DeepMatrix> uResetVectorsGradient = new List<DeepMatrix>();
            List<DeepMatrix> rKeys = new List<DeepMatrix>();
            List<DeepMatrix> rKeysGradient = new List<DeepMatrix>();
            List<DeepMatrix> rKB = new List<DeepMatrix>();
            List<DeepMatrix> rKBGradient = new List<DeepMatrix>();
            List<DeepMatrix> iKeys = new List<DeepMatrix>();
            List<DeepMatrix> iKeysGradient = new List<DeepMatrix>();
            List<DeepMatrix> iKB = new List<DeepMatrix>();
            List<DeepMatrix> iKBGradient = new List<DeepMatrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wResetWeights.Add(layer.WeightDeepMatrix("WResetWeights"));
                wResetWeightsGradient.Add(layer.GradientDeepMatrix("WResetWeights"));
                uResetWeights.Add(layer.WeightDeepMatrix("UResetWeights"));
                uResetWeightsGradient.Add(layer.GradientDeepMatrix("UResetWeights"));
                wResetVectors.Add(layer.WeightDeepMatrix("WResetVectors"));
                wResetVectorsGradient.Add(layer.GradientDeepMatrix("WResetVectors"));
                uResetVectors.Add(layer.WeightDeepMatrix("UResetVectors"));
                uResetVectorsGradient.Add(layer.GradientDeepMatrix("UResetVectors"));
                rKeys.Add(layer.WeightDeepMatrix("RKeys"));
                rKeysGradient.Add(layer.GradientDeepMatrix("RKeys"));
                rKB.Add(layer.WeightDeepMatrix("RKB"));
                rKBGradient.Add(layer.GradientDeepMatrix("RKB"));
                iKeys.Add(layer.WeightDeepMatrix("IKeys"));
                iKeysGradient.Add(layer.GradientDeepMatrix("IKeys"));
                iKB.Add(layer.WeightDeepMatrix("IKB"));
                iKBGradient.Add(layer.GradientDeepMatrix("IKB"));
            }

            List<DeepMatrix> uCWeights = new List<DeepMatrix>();
            List<DeepMatrix> uCWeightsGradient = new List<DeepMatrix>();
            List<DeepMatrix> chKeys = new List<DeepMatrix>();
            List<DeepMatrix> chKeysGradient = new List<DeepMatrix>();
            List<DeepMatrix> chKB = new List<DeepMatrix>();
            List<DeepMatrix> chKBGradient = new List<DeepMatrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                uCWeights.Add(layer.WeightDeepMatrix("UCWeights"));
                uCWeightsGradient.Add(layer.GradientDeepMatrix("UCWeights"));
                chKeys.Add(layer.WeightDeepMatrix("CHKeys"));
                chKeysGradient.Add(layer.GradientDeepMatrix("CHKeys"));
                chKB.Add(layer.WeightDeepMatrix("CHKB"));
                chKBGradient.Add(layer.GradientDeepMatrix("CHKB"));
            }

            var rowSumWeights = this.outputLayer.WeightMatrix("RowSumWeights");
            var rowSumWeightsGradient = this.outputLayer.GradientMatrix("RowSumWeights");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new SpatialComputationGraph(this);
            var zeroMatrixHiddenState = new Matrix(this.NumNodes, this.NumFeatures * 2);
            this.computationGraph
                .AddIntermediate("Output", x => this.Output[x.TimeStep])
                .AddIntermediate("Input", x => this.Input[x.TimeStep])
                .AddWeight("Weights", x => weights).AddGradient("DWeights", x => weightsGradient)
                .AddWeight("Angles", x => angles).AddGradient("DAngles", x => anglesGradient)
                .AddWeight("Vectors", x => vectors).AddGradient("DVectors", x => vectorsGradient)
                .AddWeight("WUpdateWeights", x => wUpdateWeights[x.Layer][x.TimeStep]).AddGradient("DWUpdateWeights", x => wUpdateWeightsGradient[x.Layer][x.TimeStep])
                .AddWeight("UUpdateWeights", x => uUpdateWeights[x.Layer][x.TimeStep]).AddGradient("DUUpdateWeights", x => uUpdateWeightsGradient[x.Layer][x.TimeStep])
                .AddWeight("WUpdateVectors", x => wUpdateVectors[x.Layer][x.TimeStep]).AddGradient("DWUpdateVectors", x => wUpdateVectorsGradient[x.Layer][x.TimeStep])
                .AddWeight("UUpdateVectors", x => uUpdateVectors[x.Layer][x.TimeStep]).AddGradient("DUUpdateVectors", x => uUpdateVectorsGradient[x.Layer][x.TimeStep])
                .AddWeight("ZKeys", x => zKeys[x.Layer][x.TimeStep]).AddGradient("DZKeys", x => zKeysGradient[x.Layer][x.TimeStep])
                .AddWeight("ZKB", x => zKB[x.Layer][x.TimeStep]).AddGradient("DZKB", x => zKBGradient[x.Layer][x.TimeStep])
                .AddWeight("WResetWeights", x => wResetWeights[x.Layer][x.TimeStep]).AddGradient("DWResetWeights", x => wResetWeightsGradient[x.Layer][x.TimeStep])
                .AddWeight("UResetWeights", x => uResetWeights[x.Layer][x.TimeStep]).AddGradient("DUResetWeights", x => uResetWeightsGradient[x.Layer][x.TimeStep])
                .AddWeight("WResetVectors", x => wResetVectors[x.Layer][x.TimeStep]).AddGradient("DWResetVectors", x => wResetVectorsGradient[x.Layer][x.TimeStep])
                .AddWeight("UResetVectors", x => uResetVectors[x.Layer][x.TimeStep]).AddGradient("DUResetVectors", x => uResetVectorsGradient[x.Layer][x.TimeStep])
                .AddWeight("RKeys", x => rKeys[x.Layer][x.TimeStep]).AddGradient("DRKeys", x => rKeysGradient[x.Layer][x.TimeStep])
                .AddWeight("RKB", x => rKB[x.Layer][x.TimeStep]).AddGradient("DRKB", x => rKBGradient[x.Layer][x.TimeStep])
                .AddWeight("IKeys", x => iKeys[x.Layer][x.TimeStep]).AddGradient("DIKeys", x => iKeysGradient[x.Layer][x.TimeStep])
                .AddWeight("IKB", x => iKB[x.Layer][x.TimeStep]).AddGradient("DIKB", x => iKBGradient[x.Layer][x.TimeStep])
                .AddWeight("UCWeights", x => uCWeights[x.Layer][x.TimeStep]).AddGradient("DUCWeights", x => uCWeightsGradient[x.Layer][x.TimeStep])
                .AddWeight("CHKeys", x => chKeys[x.Layer][x.TimeStep]).AddGradient("DCHKeys", x => chKeysGradient[x.Layer][x.TimeStep])
                .AddWeight("CHKB", x => chKB[x.Layer][x.TimeStep]).AddGradient("DCHKB", x => chKBGradient[x.Layer][x.TimeStep])
                .AddWeight("RowSumWeights", x => rowSumWeights).AddGradient("DRowSumWeights", x => rowSumWeightsGradient)
                .AddOperationFinder("newHFromLastLayer", x => this.computationGraph[$"compute_new_hidden_state_{x.TimeStep}_{this.NumLayers - 1}"])
                .AddOperationFinder("newHFromFirstLayer", x => this.computationGraph[$"compute_new_hidden_state_{x.TimeStep}_0"])
                .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph[$"input_projection_{x.TimeStep}_0"] : this.computationGraph[$"compute_new_hidden_state_{x.TimeStep}_{x.Layer - 1}"])
                .AddOperationFinder("previousHiddenState", x => x.TimeStep == 0 ? zeroMatrixHiddenState : this.computationGraph[$"compute_new_hidden_state_{x.TimeStep - 1}_{x.Layer}"])
                .ConstructFromArchitecture(jsonArchitecture, this.Parameters.NumTimeSteps, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph[$"output_{this.Parameters.NumTimeSteps - 1}_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, this.Parameters.NumTimeSteps - 1);
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
