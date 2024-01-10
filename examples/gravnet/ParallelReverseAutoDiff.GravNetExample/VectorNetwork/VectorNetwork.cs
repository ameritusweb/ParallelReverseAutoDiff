namespace ParallelReverseAutoDiff.GravNetExample.VectorNetwork
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GravNetExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    public class VectorNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.GravNetExample.VectorNetwork.Architecture";
        private const string ARCHITECTURE = "vectornet";

        private readonly IModelLayer inputLayer;
        private readonly List<IModelLayer> nestedLayers;
        private readonly IModelLayer outputLayer;

        private VectorComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="VectorNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public VectorNetwork(int numLayers, int numNodes, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumNodes = numNodes;
            this.NumFeatures = numFeatures;

            int numInputOutputFeatures = this.NumFeatures;
            var inputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("StartWeights", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("StartDistances", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("DivisorMatrix", new[] { 1, 1 }, InitializationType.Xavier)
                .AddModelElementGroup("KB", new[] { 1, numInputOutputFeatures }, InitializationType.Xavier);
            var inputLayer = inputLayerBuilder.Build();
            this.inputLayer = inputLayer;

            this.nestedLayers = new List<IModelLayer>();
            int numNestedOutputFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("HiddenDistances", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("HiddenDivisorMatrix", new[] { 1, 1 }, InitializationType.Xavier)
                    .AddModelElementGroup("QB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier);
                var nestedLayer = nestedLayerBuilder.Build();
                this.nestedLayers.Add(nestedLayer);
            }

            int numOutputFeatures = this.NumFeatures;
            var outputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("FW", new[] { numOutputFeatures, numOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("FB", new[] { 1, numOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("F2W", new[] { numOutputFeatures, numOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("F2B", new[] { 1, numOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("Beta", new[] { 1, 1 }, InitializationType.He);
            var outputLayer = outputLayerBuilder.Build();
            this.outputLayer = outputLayer;

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input matrix.
        /// </summary>
        public Matrix Input { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public Matrix Output { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public Matrix OutputTwo { get; private set; }

        /// <summary>
        /// Gets the model layers of the neural network.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return (new IModelLayer[] { this.inputLayer }).Concat(this.nestedLayers).Append(this.outputLayer);
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
        public void AutomaticForwardPropagate(Matrix input)
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

                if (op.Id == "deep_concatenate")
                {
                    var objArray = parameters[0] as object[] ?? throw new InvalidOperationException("Array should not be null.");
                    Matrix[] matrixArray = new Matrix[objArray.Length];
                    for (int i = 0; i < objArray.Length; ++i)
                    {
                        var obj = objArray[i];
                        if (obj is Matrix m)
                        {
                            matrixArray[i] = m;
                        }
                    }

                    parameters[0] = new DeepMatrix(matrixArray);
                }

                if (op.Id == "output")
                {

                }

                var forward = op.OperationType.GetMethod("Forward", parameters.Select(x => x.GetType()).ToArray());
                if (forward == null)
                {
                    throw new Exception($"Forward method not found for operation {op.OperationType.Name}");
                }

                if (op is VectorInfluenceOnWeightsOperation grav)
                {
                    grav.Forward(parameters[0] as Matrix, parameters[1] as Matrix, parameters[2] as Matrix, (double)parameters[3]);
                }

                forward.Invoke(op, parameters);
                var output = op.GetOutput();
                var deepOutput = op.GetDeepOutput();
                if (output != null)
                {
                    if (op.Id == "output")
                    {

                    }
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

            IOperationBase? backwardEndOperation = this.computationGraph["node_features_transform_0_0"];
            if (backwardEndOperation.CalculatedGradient[1] == null)
            {
                return gradient;
            }

            return backwardEndOperation.CalculatedGradient[1] as Matrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initialize the state of the edge attention neural network.
        /// </summary>
        public void InitializeState()
        {
            // Clear intermediates
            var output = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(this.NumNodes, this.NumFeatures).ToArray());
            var input = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(this.NumNodes, this.NumFeatures).ToArray());

            if (this.Output == null)
            {
                this.Output = output;
            }
            else
            {
                this.Output.Replace(output.ToArray());
            }

            if (this.Input == null)
            {
                this.Input = input;
            }
            else
            {
                this.Input.Replace(input.ToArray());
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
            var startWeights = this.inputLayer.WeightMatrix("StartWeights");
            var startWeightsGradient = this.inputLayer.GradientMatrix("StartWeights");

            var startDistances = this.inputLayer.WeightMatrix("StartDistances");
            var startDistancesGradient = this.inputLayer.GradientMatrix("StartDistances");

            var divisorMatrix = this.inputLayer.WeightMatrix("DivisorMatrix");
            var divisorMatrixGradient = this.inputLayer.GradientMatrix("DivisorMatrix");

            var kb = this.inputLayer.WeightMatrix("KB");
            var kbGradient = this.inputLayer.GradientMatrix("KB");

            List<Matrix> hiddenDistances = new List<Matrix>();
            List<Matrix> hiddenDivisorMatrix = new List<Matrix>();
            List<Matrix> qb = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                hiddenDistances.Add(this.nestedLayers[i].WeightMatrix("HiddenDistances"));
                hiddenDivisorMatrix.Add(this.nestedLayers[i].WeightMatrix("HiddenDivisorMatrix"));
                qb.Add(this.nestedLayers[i].WeightMatrix("QB"));
            }

            List<Matrix> hiddenDistancesGradient = new List<Matrix>();
            List<Matrix> hiddenDivisorMatrixGradient = new List<Matrix>();
            List<Matrix> qbGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                hiddenDistancesGradient.Add(this.nestedLayers[i].GradientMatrix("HiddenDistances"));
                hiddenDivisorMatrixGradient.Add(this.nestedLayers[i].GradientMatrix("HiddenDivisorMatrix"));
                qbGradient.Add(this.nestedLayers[i].GradientMatrix("QB"));
            }

            var fw = this.outputLayer.WeightMatrix("FW");
            var fwGradient = this.outputLayer.GradientMatrix("FW");

            var fb = this.outputLayer.WeightMatrix("FB");
            var fbGradient = this.outputLayer.GradientMatrix("FB");

            var f2w = this.outputLayer.WeightMatrix("F2W");
            var f2wGradient = this.outputLayer.GradientMatrix("F2W");

            var f2b = this.outputLayer.WeightMatrix("F2B");
            var f2bGradient = this.outputLayer.GradientMatrix("F2B");

            var beta = this.outputLayer.WeightMatrix("Beta");
            var betaGradient = this.outputLayer.GradientMatrix("Beta");

            double[] gravConsts = new double[] { 1.0d, 1.0d, 1.0d };

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new VectorComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("Input", _ => this.Input)
                .AddScalar("GravConst", x => 1d)
                .AddScalar("HiddenGravConst", x => gravConsts[x.Layer])
                .AddWeight("StartWeights", x => startWeights).AddGradient("DStartWeights", x => startWeightsGradient)
                .AddWeight("StartDistances", x => startDistances).AddGradient("DStartDistances", x => startDistancesGradient)
                .AddWeight("DivisorMatrix", x => divisorMatrix).AddGradient("DDivisorMatrix", x => divisorMatrixGradient)
                .AddWeight("KB", x => kb).AddGradient("DKB", x => kbGradient)
                .AddWeight("HiddenDistances", x => hiddenDistances[x.Layer]).AddGradient("DHiddenDistances", x => hiddenDistancesGradient[x.Layer])
                .AddWeight("HiddenDivisorMatrix", x => hiddenDivisorMatrix[x.Layer]).AddGradient("DHiddenDivisorMatrix", x => hiddenDivisorMatrixGradient[x.Layer])
                .AddWeight("QB", x => qb[x.Layer]).AddGradient("DQB", x => qbGradient[x.Layer])
                .AddWeight("FW", x => fw).AddGradient("DFW", x => fwGradient)
                .AddWeight("FB", x => fb).AddGradient("DFB", x => fbGradient)
                .AddWeight("F2W", x => f2w).AddGradient("DF2W", x => f2wGradient)
                .AddWeight("F2B", x => f2b).AddGradient("DF2B", x => f2bGradient)
                .AddWeight("Beta", x => beta).AddGradient("DBeta", x => betaGradient)
                .AddOperationFinder("features_act_last", x => x.Layer == 0 ? this.computationGraph[$"features_act_0_0"] : this.computationGraph[$"hidden_act_0_{x.Layer - 1}"])
                .AddOperationFinder("hidden_act_last", _ => this.computationGraph[$"hidden_act_0_{this.NumLayers - 1}"])
                .ConstructFromArchitecture(jsonArchitecture, this.NumLayers, this.NumLayers);

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
