namespace ParallelReverseAutoDiff.GravNetExample.VectorFieldNetwork
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GravNetExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    public class VectorFieldNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.GravNetExample.VectorFieldNetwork.Architecture";
        private const string ARCHITECTURE = "vectorfieldnet";

        private readonly IModelLayer inputLayer;

        private VectorFieldComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="VectorFieldNetwork"/> class.
        /// </summary>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public VectorFieldNetwork(int numLayers, int numNodes, int numFeatures, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.NumLayers = numLayers;
            this.NumNodes = numNodes;
            this.NumFeatures = numFeatures;

            int numInputOutputFeatures = this.NumFeatures;
            var initial = this.NumFeatures / 10;
            var inputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("Angles", new[] { numNodes, initial / 2 }, InitializationType.Xavier)
                .AddModelElementGroup("ProjectionVectors", new[] { numNodes, numInputOutputFeatures / 2 }, InitializationType.Xavier)
                .AddModelElementGroup("ProjectionWeights", new[] { numNodes, initial / 2 }, InitializationType.HeAdjacency)
                .AddModelElementGroup("WeightVectors", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("WeightVectors2", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("Weights", new[] { numInputOutputFeatures / 2, numInputOutputFeatures / 2 }, InitializationType.Xavier)
                .AddModelElementGroup("Weights2", new[] { numInputOutputFeatures / 2, numInputOutputFeatures / 2 }, InitializationType.Xavier)
                .AddModelElementGroup("Keys", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("KB", new[] { 1, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("Queries", new[] { numInputOutputFeatures, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("QB", new[] { 1, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("SummationWeights", new[] { numNodes, numInputOutputFeatures / 2 }, InitializationType.Xavier);
            var inputLayer = inputLayerBuilder.Build();
            this.inputLayer = inputLayer;

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

        public double TargetAngle { get; set; }

        public double OppositeAngle { get; set; }

        public int[] GoodTotals { get; set; }

        public int[] BadTotals { get; set; }

        public int[] OnlyGoodTotals { get; set; }

        public int[] OnlyBadTotals { get; set; }

        public List<(List<int>, List<int>)> Rankings { get; set; }

        public List<int> LastGoodIndices { get; set; }

        /// <summary>
        /// Gets the model layers of the neural network.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return (new[] { this.inputLayer });
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

        public void RecordVectors(double[,] vectors)
        {

            if (this.GoodTotals == null)
            {
                this.GoodTotals = new int[vectors.GetLength(0)];
            }

            if (this.BadTotals == null)
            {
                this.BadTotals = new int[vectors.GetLength(0)];
            }

            if (this.OnlyGoodTotals == null)
            {
                this.OnlyGoodTotals = new int[vectors.GetLength(0)];
            }

            if (this.OnlyBadTotals == null)
            {
                this.OnlyBadTotals = new int[vectors.GetLength(0)];
            }

            if (this.Rankings == null)
            {
                this.Rankings = new List<(List<int>, List<int>)>();
            }

            var targetAngle = this.TargetAngle;
            var oppositeAngle = this.OppositeAngle;
            bool[] goodDiffs = new bool[vectors.GetLength(0)];
            double magnitudeTotal = 0d;
            double[] magnitudes = new double[vectors.GetLength(0)];
            for (int i = 0; i < vectors.GetLength(0); i++)
            {
                var vectorMagnitude = vectors[i, 0];
                magnitudes[i] = vectorMagnitude;
                magnitudeTotal += vectorMagnitude;
                var vectorAngle = vectors[i, 1];
                var angleTargetDiff = Math.Abs(vectorAngle - targetAngle);
                var angleOppositeDiff = Math.Abs(vectorAngle - oppositeAngle);
                goodDiffs[i] = angleTargetDiff > angleOppositeDiff ? false : true;

                //if (i == 67 || i == 282)
                //{
                //    double adiff = (Math.Abs(this.TargetAngle - this.OppositeAngle) / 2d);
                //    Console.WriteLine(i + ": magnitude:" + vectorMagnitude + " angle: " + vectorAngle + " target: " + this.TargetAngle + " mid: " + (this.TargetAngle > this.OppositeAngle ? this.OppositeAngle + adiff : this.TargetAngle + adiff));
                //}
            }

            var thresholdMagnitude = magnitudeTotal / vectors.GetLength(0) / 4d;

            List<int> goodIndices = new List<int>();
            List<int> badIndices = new List<int>();
            for (int i = 0; i < vectors.GetLength(0); ++i)
            {
                var magnitude = magnitudes[i];
                if (magnitude > thresholdMagnitude)
                {
                    if (goodDiffs[i])
                    {
                        goodIndices.Add(i);
                        this.GoodTotals[i]++;
                        if (this.OnlyBadTotals[i] > 0)
                        {
                            this.OnlyBadTotals[i] = -1;
                        }

                        if (this.OnlyGoodTotals[i] != -1)
                        {
                            this.OnlyGoodTotals[i]++;
                        }
                    } else
                    {
                        badIndices.Add(i);
                        this.BadTotals[i]++;
                        if (this.OnlyGoodTotals[i] > 0)
                        {
                            this.OnlyGoodTotals[i] = -1;
                        }

                        if (this.OnlyBadTotals[i] != -1)
                        {
                            this.OnlyBadTotals[i]++;
                        }
                    }
                }
            }

            LastGoodIndices = goodIndices.ToList();

            List<int> onlyGoodIndices = new List<int>();
            List<int> onlyBadIndices = new List<int>();
            for (int i = 0; i < vectors.GetLength(0); ++i)
            {
                if (this.OnlyGoodTotals[i] > 0)
                {
                    onlyGoodIndices.Add(i);
                }

                if (this.OnlyBadTotals[i] > 0)
                {
                    onlyBadIndices.Add(i);
                }
            }

            if (onlyGoodIndices.Count > 0 || onlyBadIndices.Count > 0)
            {
                this.Rankings.Add((onlyGoodIndices.ToList(), onlyBadIndices.ToList()));
            }

         }

        public (double, double) SumVectors(List<int> indices, double[,] vectors)
        {
            double sumX = 0d;
            double sumY = 0d;
            foreach (var index in indices)
            {
                var polarMagnitude = vectors[index, 0];
                var polarAngle = vectors[index, 1];
                var x = polarMagnitude * Math.Cos(polarAngle);
                var y = polarMagnitude * Math.Sin(polarAngle);
                sumX += x;
                sumY += y;
            }

            double combinedPolarMagnitude = Math.Sqrt((sumX * sumX) + (sumY * sumY));
            double combinedPolarAngle = Math.Atan2(sumY, sumX);

            return (combinedPolarMagnitude, combinedPolarAngle);
        }

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

            IOperationBase? backwardEndOperation = this.computationGraph["weight_vectors_square_0_0"];
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
            var output = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(1, 2).ToArray());
            var input = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(this.NumNodes, this.NumFeatures / 2).ToArray());

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
            var angles = this.inputLayer.WeightMatrix("Angles");
            var anglesGradient = this.inputLayer.GradientMatrix("Angles");

            var projectionVectors = this.inputLayer.WeightMatrix("ProjectionVectors");
            var projectionVectorsGradient = this.inputLayer.GradientMatrix("ProjectionVectors");

            var projectionWeights = this.inputLayer.WeightMatrix("ProjectionWeights");
            var projectionWeightsGradient = this.inputLayer.GradientMatrix("ProjectionWeights");

            var weightVectors = this.inputLayer.WeightMatrix("WeightVectors");
            var weightVectorsGradient = this.inputLayer.GradientMatrix("WeightVectors");

            var weightVectors2 = this.inputLayer.WeightMatrix("WeightVectors2");
            var weightVectors2Gradient = this.inputLayer.GradientMatrix("WeightVectors2");

            var weights = this.inputLayer.WeightMatrix("Weights");
            var weightsGradient = this.inputLayer.GradientMatrix("Weights");

            var weights2 = this.inputLayer.WeightMatrix("Weights2");
            var weights2Gradient = this.inputLayer.GradientMatrix("Weights2");

            var keys = this.inputLayer.WeightMatrix("Keys");
            var keysGradient = this.inputLayer.GradientMatrix("Keys");

            var kb = this.inputLayer.WeightMatrix("KB");
            var kbGradient = this.inputLayer.GradientMatrix("KB");

            var queries = this.inputLayer.WeightMatrix("Queries");
            var queriesGradient = this.inputLayer.GradientMatrix("Queries");

            var qb = this.inputLayer.WeightMatrix("QB");
            var qbGradient = this.inputLayer.GradientMatrix("QB");

            var summationWeights = this.inputLayer.WeightMatrix("SummationWeights");
            var summationWeightsGradient = this.inputLayer.GradientMatrix("SummationWeights");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new VectorFieldComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("Input", _ => this.Input)
                .AddWeight("Angles", x => angles).AddGradient("DAngles", x => anglesGradient)
                .AddWeight("ProjectionVectors", x => projectionVectors).AddGradient("DProjectionVectors", x => projectionVectorsGradient)
                .AddWeight("ProjectionWeights", x => projectionWeights).AddGradient("DProjectionWeights", x => projectionWeightsGradient)
                .AddWeight("WeightVectors", x => weightVectors).AddGradient("DWeightVectors", x => weightVectorsGradient)
                .AddWeight("WeightVectors2", x => weightVectors2).AddGradient("DWeightVectors2", x => weightVectors2Gradient)
                .AddWeight("Weights", x => weights).AddGradient("DWeights", x => weightsGradient)
                .AddWeight("Weights2", x => weights2).AddGradient("DWeights2", x => weights2Gradient)
                .AddWeight("Keys", x => keys).AddGradient("DKeys", x => keysGradient)
                .AddWeight("KB", x => kb).AddGradient("DKB", x => kbGradient)
                .AddWeight("Queries", x => queries).AddGradient("DQueries", x => queriesGradient)
                .AddWeight("QB", x => qb).AddGradient("DQB", x => qbGradient)
                .AddWeight("SummationWeights", x => summationWeights).AddGradient("DSummationWeights", x => summationWeightsGradient)
                .ConstructFromArchitecture(jsonArchitecture);

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
