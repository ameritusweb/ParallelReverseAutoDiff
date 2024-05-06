// ------------------------------------------------------------------------------
// <copyright file="MusicNetwork.cs" author="ameritusweb" date="12/18/2023">
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
    public class MusicNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.VGruExample.VGruNetwork.Architecture";
        private const string ARCHITECTURE = "musicnet";

        private readonly IModelLayer inputLayer;
        private readonly List<IModelLayer> nestedLayers;
        private readonly IModelLayer outputLayer;

        private MusicComputationGraph computationGraph;

        private Maze maze;

        /// <summary>
        /// Initializes a new instance of the <see cref="MusicNetwork"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public MusicNetwork(int numTimeSteps, int numLayers, int numNodes, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumNodes = numNodes;
            this.NumFeatures = numFeatures;
            this.Parameters.NumTimeSteps = numTimeSteps;

            this.maze = new Maze();

            int numInputOutputFeatures = this.NumFeatures;
            var inputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Weights", new[] { numTimeSteps, numNodes, numInputOutputFeatures / 10 }, InitializationType.XavierUniform, 1.0d)
                .AddModelElementGroup("Angles", new[] { numTimeSteps, numInputOutputFeatures / 10, numInputOutputFeatures / 10 }, InitializationType.XavierUniform, 1.0d)
                .AddModelElementGroup("Vectors", new[] { numTimeSteps, numNodes, numInputOutputFeatures }, InitializationType.XavierUniform, 1.0d);
            var inputLayer = inputLayerBuilder.Build();
            this.inputLayer = inputLayer;

            this.nestedLayers = new List<IModelLayer>();
            int numNestedOutputFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("UpdateWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("ResetWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("CandidateWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("HiddenWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("WUpdateWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UUpdateWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("WUpdateVectors", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UUpdateVectors", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("ZKeys", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("ZKB", new[] { numTimeSteps, 1, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("WResetWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UResetWeights", new[] { numTimeSteps, numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("WResetVectors", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
                    .AddModelElementGroup("UResetVectors", new[] { numTimeSteps, numNestedOutputFeatures * 2, numNestedOutputFeatures * 2 }, InitializationType.XavierUniform, 1.0d)
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
                .AddModelElementGroup("RowSumWeights", new[] { numTimeSteps, numNodes, numInputOutputFeatures }, InitializationType.Xavier);
            var outputLayer = outputLayerBuilder.Build();
            this.outputLayer = outputLayer;

            this.maze.SetLayers(inputLayer, this.nestedLayers.ToArray(), outputLayer);
            this.maze.InitializeStructure();
            this.maze.SetWeightsFromStructure();

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input matrices.
        /// </summary>
        public Matrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrices.
        /// </summary>
        public Matrix Output { get; private set; }

        /// <summary>
        /// Gets the hidden state.
        /// </summary>
        public DeepMatrix HiddenState { get; private set; }

        /// <summary>
        /// Gets or sets the previous hidden state.
        /// </summary>
        public Matrix PreviousHiddenState { get; set; }

        /// <summary>
        /// Gets or sets the current time step.
        /// </summary>
        public int CurrentTimeStep { get; set; }

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
        /// Gets or sets the computation graph.
        /// </summary>
        public MusicComputationGraph ComputationGraph
        {
            get
            {
                return this.computationGraph;
            }

            set
            {
                this.computationGraph = value;
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
        /// Reinitializes the maze network.
        /// </summary>
        /// <param name="structure">The structure.</param>
        /// <returns>The task.</returns>
        public async Task Reinitialize(int[,] structure)
        {
            this.maze.ReinitializeAndUpdate(structure);

            await this.InitializeComputationGraph();
        }

        /// <summary>
        /// Reset the maze.
        /// </summary>
        public void ResetMaze()
        {
            this.maze.InitializeStructure();
            this.maze.SetWeightsFromStructure();
        }

        /// <summary>
        /// Set the maze.
        /// </summary>
        /// <param name="maze">The maze.</param>
        public void SetMaze(Maze maze)
        {
            this.maze = maze;
        }

        /// <summary>
        /// Clone the maze.
        /// </summary>
        /// <returns>The cloned maze.</returns>
        public Maze CloneMaze()
        {
            return (Maze)this.maze.Clone();
        }

        /// <summary>
        /// Updates model layers.
        /// </summary>
        public void UpdateModelLayers()
        {
            this.maze.UpdateModelLayers();
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
        /// The forward pass of the maze neural network.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="previousHiddenState">The previous hidden state.</param>
        public void AutomaticForwardPropagate(Matrix input, Matrix? previousHiddenState)
        {
            CommonMatrixUtils.SetInPlaceReplace(this.Input, input);
            if (previousHiddenState != null)
            {
                CommonMatrixUtils.SetInPlaceReplace(this.PreviousHiddenState, previousHiddenState);
            }

            var op = this.computationGraph.StartOperation;
            if (op == null)
            {
                throw new Exception("Start operation should not be null.");
            }

            IOperationBase? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);

                for (int i = 0; i < parameters.Length; ++i)
                {
                    if (parameters[i] is Matrix m)
                    {
                        parameters[i] = m.Clone();
                    }
                    else
                    {
                        throw new InvalidOperationException("Parameters should be matrices.");
                    }
                }

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
        /// <param name="computationGraph">The computation graph.</param>
        /// <returns>The gradient.</returns>
        public async Task<Matrix> AutomaticBackwardPropagate(Matrix gradient, MusicComputationGraph computationGraph)
        {
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = computationGraph[$"output_0_0"];

            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = true;
                var op1 = computationGraph["z_add_0_0"];
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

            IOperationBase? backwardEndOperation = computationGraph["scale_previous_by_modulated_z_0_0"];
            if (backwardEndOperation.CalculatedGradient[0] != null)
            {
                return gradient;
            }

            return backwardEndOperation.CalculatedGradient[0] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initialize the state of the edge attention neural network.
        /// </summary>
        /// <param name="inputMatrix">The input matrix.</param>
        public void InitializeState(Matrix? inputMatrix = null)
        {
            if (inputMatrix != null)
            {
                this.NumNodes = inputMatrix.Rows;
                this.NumFeatures = inputMatrix.Cols * 10;
            }

            // Clear intermediates
            var output = CommonMatrixUtils.InitializeZeroMatrix(1, 2);
            var input = CommonMatrixUtils.InitializeZeroMatrix(this.NumNodes, this.NumFeatures);
            var hiddenState = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(2, this.NumNodes, this.NumFeatures * 2));
            var previousHiddenState = CommonMatrixUtils.InitializeZeroMatrix(this.NumNodes, this.NumFeatures * 2);

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

            if (this.HiddenState == null)
            {
                this.HiddenState = hiddenState;
            }
            else
            {
                CommonMatrixUtils.SetInPlaceReplace(this.HiddenState, hiddenState);
            }

            if (this.PreviousHiddenState == null)
            {
                this.PreviousHiddenState = previousHiddenState;
            }
            else
            {
                CommonMatrixUtils.SetInPlaceReplace(this.PreviousHiddenState, previousHiddenState);
            }
        }

        /// <summary>
        /// Clear the state of the edge attention neural network.
        /// </summary>
        public void ClearState()
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
            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new MusicComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", x => this.Output)
                .AddIntermediate("Input", x => this.Input)
                .AddIntermediate("HiddenState", x => this.HiddenState[x.Layer])
                .AddIntermediate("previousHiddenState", x => this.PreviousHiddenState)
                .AddWeight("Weights", x => this.maze.Weights[0]).AddGradient("DWeights", x => this.maze.Weights[1])
                .AddWeight("Angles", x => this.maze.Angles[0]).AddGradient("DAngles", x => this.maze.Angles[1])
                .AddWeight("Vectors", x => this.maze.Vectors[0]).AddGradient("DVectors", x => this.maze.Vectors[1])
                .AddWeight("UpdateWeights", x => this.maze.Layers[x.Layer].UpdateWeights[0]).AddGradient("DUpdateWeights", x => this.maze.Layers[x.Layer].UpdateWeights[1])
                .AddWeight("ResetWeights", x => this.maze.Layers[x.Layer].ResetWeights[0]).AddGradient("DResetWeights", x => this.maze.Layers[x.Layer].ResetWeights[1])
                .AddWeight("CandidateWeights", x => this.maze.Layers[x.Layer].CandidateWeights[0]).AddGradient("DCandidateWeights", x => this.maze.Layers[x.Layer].CandidateWeights[1])
                .AddWeight("HiddenWeights", x => this.maze.Layers[x.Layer].HiddenWeights[0]).AddGradient("DHiddenWeights", x => this.maze.Layers[x.Layer].HiddenWeights[1])
                .AddWeight("WUpdateWeights", x => this.maze.Layers[x.Layer].WUpdateWeights[0]).AddGradient("DWUpdateWeights", x => this.maze.Layers[x.Layer].WUpdateWeights[1])
                .AddWeight("UUpdateWeights", x => this.maze.Layers[x.Layer].UUpdateWeights[0]).AddGradient("DUUpdateWeights", x => this.maze.Layers[x.Layer].UUpdateWeights[1])
                .AddWeight("WUpdateVectors", x => this.maze.Layers[x.Layer].WUpdateVectors[0]).AddGradient("DWUpdateVectors", x => this.maze.Layers[x.Layer].WUpdateVectors[1])
                .AddWeight("UUpdateVectors", x => this.maze.Layers[x.Layer].UUpdateVectors[0]).AddGradient("DUUpdateVectors", x => this.maze.Layers[x.Layer].UUpdateVectors[1])
                .AddWeight("ZKeys", x => this.maze.Layers[x.Layer].ZKeys[0]).AddGradient("DZKeys", x => this.maze.Layers[x.Layer].ZKeys[1])
                .AddWeight("ZKB", x => this.maze.Layers[x.Layer].ZKB[0]).AddGradient("DZKB", x => this.maze.Layers[x.Layer].ZKB[1])
                .AddWeight("WResetWeights", x => this.maze.Layers[x.Layer].WResetWeights[0]).AddGradient("DWResetWeights", x => this.maze.Layers[x.Layer].WResetWeights[1])
                .AddWeight("UResetWeights", x => this.maze.Layers[x.Layer].UResetWeights[0]).AddGradient("DUResetWeights", x => this.maze.Layers[x.Layer].UResetWeights[1])
                .AddWeight("WResetVectors", x => this.maze.Layers[x.Layer].WResetVectors[0]).AddGradient("DWResetVectors", x => this.maze.Layers[x.Layer].WResetVectors[1])
                .AddWeight("UResetVectors", x => this.maze.Layers[x.Layer].UResetVectors[0]).AddGradient("DUResetVectors", x => this.maze.Layers[x.Layer].UResetVectors[1])
                .AddWeight("RKeys", x => this.maze.Layers[x.Layer].RKeys[0]).AddGradient("DRKeys", x => this.maze.Layers[x.Layer].RKeys[1])
                .AddWeight("RKB", x => this.maze.Layers[x.Layer].RKB[0]).AddGradient("DRKB", x => this.maze.Layers[x.Layer].RKB[1])
                .AddWeight("IKeys", x => this.maze.Layers[x.Layer].IKeys[0]).AddGradient("DIKeys", x => this.maze.Layers[x.Layer].IKeys[1])
                .AddWeight("IKB", x => this.maze.Layers[x.Layer].IKB[0]).AddGradient("DIKB", x => this.maze.Layers[x.Layer].IKB[1])
                .AddWeight("UCWeights", x => this.maze.Layers[x.Layer].UCWeights[0]).AddGradient("DUCWeights", x => this.maze.Layers[x.Layer].UCWeights[1])
                .AddWeight("CHKeys", x => this.maze.Layers[x.Layer].CHKeys[0]).AddGradient("DCHKeys", x => this.maze.Layers[x.Layer].CHKeys[1])
                .AddWeight("CHKB", x => this.maze.Layers[x.Layer].CHKB[0]).AddGradient("DCHKB", x => this.maze.Layers[x.Layer].CHKB[1])
                .AddWeight("RowSumWeights", x => this.maze.RowSumWeights[0]).AddGradient("DRowSumWeights", x => this.maze.RowSumWeights[1])
                .AddOperationFinder("newHFromLastLayer", x => this.computationGraph[$"compute_new_hidden_state_0_{this.NumLayers - 1}"])
                .AddOperationFinder("newHFromFirstLayer", x => this.computationGraph[$"compute_new_hidden_state_0_0"])
                .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph[$"input_projection_0_0"] : this.computationGraph[$"compute_new_hidden_state_0_{x.Layer - 1}"])
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph[$"output_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
