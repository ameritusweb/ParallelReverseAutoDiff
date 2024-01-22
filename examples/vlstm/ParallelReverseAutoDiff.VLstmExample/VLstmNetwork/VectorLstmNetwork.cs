namespace ParallelReverseAutoDiff.VLstmExample.VLstmNetwork
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VLstmExample.Common;

    public class VectorLstmNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.VLstmExample.VLstmNetwork.Architecture";
        private const string ARCHITECTURE = "vlstmnet";

        private readonly IModelLayer inputLayer;
        private readonly List<IModelLayer> nestedLayers;
        private readonly IModelLayer outputLayer;

        private VectorLstmComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="VectorLstmNetwork"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public VectorLstmNetwork(int numTimeSteps, int numLayers, int numNodes, int numFeatures, double learningRate, double clipValue)
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
                .AddModelElementGroup("Vectors", new[] { numNodes, numInputOutputFeatures }, InitializationType.Xavier)
                .AddModelElementGroup("PolarWeights", new[] { numNodes, numInputOutputFeatures / 2 }, InitializationType.Xavier)
                .AddModelElementGroup("PolarVectors", new[] { numNodes, numInputOutputFeatures }, InitializationType.Xavier);
            var inputLayer = inputLayerBuilder.Build();
            this.inputLayer = inputLayer;

            this.nestedLayers = new List<IModelLayer>();
            int numNestedOutputFeatures = this.NumFeatures;
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var nestedLayerBuilder = new ModelLayerBuilder(this)
                    .AddModelElementGroup("WForgetWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("UForgetWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("WForgetVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("UForgetVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("FKeys", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("FKB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("WInputWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("UInputWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("WInputVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("UInputVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("IKeys", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("IKB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("WCWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("UCWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("WCVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("UCVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("CKeys", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("CKB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("WOutputWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("UOutputWeights", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("WOutputVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("UOutputVectors", new[] { numNestedOutputFeatures / 2, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("OKeys", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("OKB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("PreviousWeights", new[] { numNodes, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("CWeights", new[] { numNodes, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("HiddenWeights", new[] { numNodes, numNestedOutputFeatures / 2 }, InitializationType.Xavier)
                    .AddModelElementGroup("HKeys", new[] { numNestedOutputFeatures, numNestedOutputFeatures }, InitializationType.Xavier)
                    .AddModelElementGroup("HKB", new[] { 1, numNestedOutputFeatures }, InitializationType.Xavier);
                var nestedLayer = nestedLayerBuilder.Build();
                this.nestedLayers.Add(nestedLayer);
            }

            int numOutputFeatures = this.NumFeatures;
            var outputLayerBuilder = new ModelLayerBuilder(this)
                .AddModelElementGroup("RowSumWeights", new[] { numTimeSteps, numInputOutputFeatures / 2 }, InitializationType.Xavier);
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

        public Matrix SoftmaxDecision { get; private set; }

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
                return (new IModelLayer[] { this.inputLayer });
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
            var output = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(1, this.NumFeatures).ToArray());
            var input = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(1, 2).ToArray());
            var softmaxDecision = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(1, this.Parameters.NumTimeSteps).ToArray());

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

            if (this.SoftmaxDecision == null)
            {
                this.SoftmaxDecision = softmaxDecision;
            }
            else
            {
                this.SoftmaxDecision.Replace(softmaxDecision.ToArray());
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
            var polarWeights = this.inputLayer.WeightMatrix("PolarWeights");
            var polarWeightsGradient = this.inputLayer.GradientMatrix("PolarWeights");
            var polarVectors = this.inputLayer.WeightMatrix("PolarVectors");
            var polarVectorsGradient = this.inputLayer.GradientMatrix("PolarVectors");

            List<Matrix> wForgetWeights = new List<Matrix>();
            List<Matrix> wForgetWeightsGradient = new List<Matrix>();
            List<Matrix> uForgetWeights = new List<Matrix>();
            List<Matrix> uForgetWeightsGradient = new List<Matrix>();
            List<Matrix> wForgetVectors = new List<Matrix>();
            List<Matrix> wForgetVectorsGradient = new List<Matrix>();
            List<Matrix> uForgetVectors = new List<Matrix>();
            List<Matrix> uForgetVectorsGradient = new List<Matrix>();
            List<Matrix> fKeys = new List<Matrix>();
            List<Matrix> fKeysGradient = new List<Matrix>();
            List<Matrix> fKB = new List<Matrix>();
            List<Matrix> fKBGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wForgetWeights.Add(layer.WeightMatrix("WForgetWeights"));
                wForgetWeightsGradient.Add(layer.GradientMatrix("WForgetWeights"));
                uForgetWeights.Add(layer.WeightMatrix("UForgetWeights"));
                uForgetWeightsGradient.Add(layer.GradientMatrix("UForgetWeights"));
                wForgetVectors.Add(layer.WeightMatrix("WForgetVectors"));
                wForgetVectorsGradient.Add(layer.GradientMatrix("WForgetVectors"));
                uForgetVectors.Add(layer.WeightMatrix("UForgetVectors"));
                uForgetVectorsGradient.Add(layer.GradientMatrix("UForgetVectors"));
                fKeys.Add(layer.WeightMatrix("FKeys"));
                fKeysGradient.Add(layer.GradientMatrix("FKeys"));
                fKB.Add(layer.WeightMatrix("FKB"));
                fKBGradient.Add(layer.GradientMatrix("FKB"));
            }

            List<Matrix> wInputWeights = new List<Matrix>();
            List<Matrix> wInputWeightsGradient = new List<Matrix>();
            List<Matrix> uInputWeights = new List<Matrix>();
            List<Matrix> uInputWeightsGradient = new List<Matrix>();
            List<Matrix> wInputVectors = new List<Matrix>();
            List<Matrix> wInputVectorsGradient = new List<Matrix>();
            List<Matrix> uInputVectors = new List<Matrix>();
            List<Matrix> uInputVectorsGradient = new List<Matrix>();
            List<Matrix> iKeys = new List<Matrix>();
            List<Matrix> iKeysGradient = new List<Matrix>();
            List<Matrix> iKB = new List<Matrix>();
            List<Matrix> iKBGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wInputWeights.Add(layer.WeightMatrix("WInputWeights"));
                wInputWeightsGradient.Add(layer.GradientMatrix("WInputWeights"));
                uInputWeights.Add(layer.WeightMatrix("UInputWeights"));
                uInputWeightsGradient.Add(layer.GradientMatrix("UInputWeights"));
                wInputVectors.Add(layer.WeightMatrix("WInputVectors"));
                wInputVectorsGradient.Add(layer.GradientMatrix("WInputVectors"));
                uInputVectors.Add(layer.WeightMatrix("UInputVectors"));
                uInputVectorsGradient.Add(layer.GradientMatrix("UInputVectors"));
                iKeys.Add(layer.WeightMatrix("IKeys"));
                iKeysGradient.Add(layer.GradientMatrix("IKeys"));
                iKB.Add(layer.WeightMatrix("IKB"));
                iKBGradient.Add(layer.GradientMatrix("IKB"));
            }

            List<Matrix> wCWeights = new List<Matrix>();
            List<Matrix> wCWeightsGradient = new List<Matrix>();
            List<Matrix> uCWeights = new List<Matrix>();
            List<Matrix> uCWeightsGradient = new List<Matrix>();
            List<Matrix> wCVectors = new List<Matrix>();
            List<Matrix> wCVectorsGradient = new List<Matrix>();
            List<Matrix> uCVectors = new List<Matrix>();
            List<Matrix> uCVectorsGradient = new List<Matrix>();
            List<Matrix> cKeys = new List<Matrix>();
            List<Matrix> cKeysGradient = new List<Matrix>();
            List<Matrix> cKB = new List<Matrix>();
            List<Matrix> cKBGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wCWeights.Add(layer.WeightMatrix("WCWeights"));
                wCWeightsGradient.Add(layer.GradientMatrix("WCWeights"));
                uCWeights.Add(layer.WeightMatrix("UCWeights"));
                uCWeightsGradient.Add(layer.GradientMatrix("UCWeights"));
                wCVectors.Add(layer.WeightMatrix("WCVectors"));
                wCVectorsGradient.Add(layer.GradientMatrix("WCVectors"));
                uCVectors.Add(layer.WeightMatrix("UCVectors"));
                uCVectorsGradient.Add(layer.GradientMatrix("UCVectors"));
                cKeys.Add(layer.WeightMatrix("CKeys"));
                cKeysGradient.Add(layer.GradientMatrix("CKeys"));
                cKB.Add(layer.WeightMatrix("CKB"));
                cKBGradient.Add(layer.GradientMatrix("CKB"));
            }

            List<Matrix> wOutputWeights = new List<Matrix>();
            List<Matrix> wOutputWeightsGradient = new List<Matrix>();
            List<Matrix> uOutputWeights = new List<Matrix>();
            List<Matrix> uOutputWeightsGradient = new List<Matrix>();
            List<Matrix> wOutputVectors = new List<Matrix>();
            List<Matrix> wOutputVectorsGradient = new List<Matrix>();
            List<Matrix> uOutputVectors = new List<Matrix>();
            List<Matrix> uOutputVectorsGradient = new List<Matrix>();
            List<Matrix> oKeys = new List<Matrix>();
            List<Matrix> oKeysGradient = new List<Matrix>();
            List<Matrix> oKB = new List<Matrix>();
            List<Matrix> oKBGradient = new List<Matrix>();
            List<Matrix> hKeys = new List<Matrix>();
            List<Matrix> hKeysGradient = new List<Matrix>();
            List<Matrix> hKB = new List<Matrix>();
            List<Matrix> hKBGradient = new List<Matrix>();
            List<Matrix> previousWeights = new List<Matrix>();
            List<Matrix> previousWeightsGradient = new List<Matrix>();
            List<Matrix> cWeights = new List<Matrix>();
            List<Matrix> cWeightsGradient = new List<Matrix>();
            List<Matrix> hiddenWeights = new List<Matrix>();
            List<Matrix> hiddenWeightsGradient = new List<Matrix>();
            for (int i = 0; i < this.NumLayers; ++i)
            {
                var layer = this.nestedLayers[i];
                wOutputWeights.Add(layer.WeightMatrix("WOutputWeights"));
                wOutputWeightsGradient.Add(layer.GradientMatrix("WOutputWeights"));
                uOutputWeights.Add(layer.WeightMatrix("UOutputWeights"));
                uOutputWeightsGradient.Add(layer.GradientMatrix("UOutputWeights"));
                wOutputVectors.Add(layer.WeightMatrix("WOutputVectors"));
                wOutputVectorsGradient.Add(layer.GradientMatrix("WOutputVectors"));
                uOutputVectors.Add(layer.WeightMatrix("UOutputVectors"));
                uOutputVectorsGradient.Add(layer.GradientMatrix("UOutputVectors"));
                oKeys.Add(layer.WeightMatrix("OKeys"));
                oKeysGradient.Add(layer.GradientMatrix("OKeys"));
                oKB.Add(layer.WeightMatrix("OKB"));
                oKBGradient.Add(layer.GradientMatrix("OKB"));
                hKeys.Add(layer.WeightMatrix("HKeys"));
                hKeysGradient.Add(layer.GradientMatrix("HKeys"));
                hKB.Add(layer.WeightMatrix("HKB"));
                hKBGradient.Add(layer.GradientMatrix("HKB"));
                previousWeights.Add(layer.WeightMatrix("PreviousWeights"));
                previousWeightsGradient.Add(layer.GradientMatrix("PreviousWeights"));
                cWeights.Add(layer.WeightMatrix("CWeights"));
                cWeightsGradient.Add(layer.GradientMatrix("CWeights"));
                hiddenWeights.Add(layer.WeightMatrix("HiddenWeights"));
                hiddenWeightsGradient.Add(layer.GradientMatrix("HiddenWeights"));
            }

            var rowSumWeights = this.outputLayer.WeightMatrix("RowSumWeights");
            var rowSumWeightsGradient = this.inputLayer.GradientMatrix("RowSumWeights");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new VectorLstmComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("Input", _ => this.Input)
                .AddIntermediate("SoftmaxDecision", _ => this.SoftmaxDecision)
                .AddWeight("Weights", x => weights).AddGradient("DWeights", x => weightsGradient)
                .AddWeight("Vectors", x => vectors).AddGradient("DVectors", x => vectorsGradient)
                .AddWeight("PolarWeights", x => polarWeights).AddGradient("DPolarWeights", x => polarWeightsGradient)
                .AddWeight("PolarVectors", x => polarVectors).AddGradient("DPolarVectors", x => polarVectorsGradient)
                .AddWeight("WForgetWeights", x => wForgetWeights[x.Layer]).AddGradient("DWForgetWeights", x => wForgetWeightsGradient[x.Layer])
                .AddWeight("UForgetWeights", x => uForgetWeights[x.Layer]).AddGradient("DUForgetWeights", x => uForgetWeightsGradient[x.Layer])
                .AddWeight("WForgetVectors", x => wForgetVectors[x.Layer]).AddGradient("DWForgetVectors", x => wForgetVectorsGradient[x.Layer])
                .AddWeight("UForgetVectors", x => uForgetVectors[x.Layer]).AddGradient("DUForgetVectors", x => uForgetVectorsGradient[x.Layer])
                .AddWeight("FKeys", x => fKeys[x.Layer]).AddGradient("DFKeys", x => fKeysGradient[x.Layer])
                .AddWeight("FKB", x => fKB[x.Layer]).AddGradient("DFKB", x => fKBGradient[x.Layer])
                .AddWeight("WInputWeights", x => wInputWeights[x.Layer]).AddGradient("DWInputWeights", x => wInputWeightsGradient[x.Layer])
                .AddWeight("UInputWeights", x => uInputWeights[x.Layer]).AddGradient("DUInputWeights", x => uInputWeightsGradient[x.Layer])
                .AddWeight("WInputVectors", x => wInputVectors[x.Layer]).AddGradient("DWInputVectors", x => wInputVectorsGradient[x.Layer])
                .AddWeight("UInputVectors", x => uInputVectors[x.Layer]).AddGradient("DUInputVectors", x => uInputVectorsGradient[x.Layer])
                .AddWeight("IKeys", x => iKeys[x.Layer]).AddGradient("DIKeys", x => iKeysGradient[x.Layer])
                .AddWeight("IKB", x => iKB[x.Layer]).AddGradient("DIKB", x => iKBGradient[x.Layer])
                .AddWeight("WCWeights", x => wCWeights[x.Layer]).AddGradient("DWCWeights", x => wCWeightsGradient[x.Layer])
                .AddWeight("UCWeights", x => uCWeights[x.Layer]).AddGradient("DUCWeights", x => uCWeightsGradient[x.Layer])
                .AddWeight("WCVectors", x => wCVectors[x.Layer]).AddGradient("DWCVectors", x => wCVectorsGradient[x.Layer])
                .AddWeight("UCVectors", x => uCVectors[x.Layer]).AddGradient("DUCVectors", x => uCVectorsGradient[x.Layer])
                .AddWeight("CKeys", x => cKeys[x.Layer]).AddGradient("DCKeys", x => cKeysGradient[x.Layer])
                .AddWeight("CKB", x => cKB[x.Layer]).AddGradient("DCKB", x => cKBGradient[x.Layer])
                .AddWeight("WOutputWeights", x => wOutputWeights[x.Layer]).AddGradient("DWOutputWeights", x => wOutputWeightsGradient[x.Layer])
                .AddWeight("UOutputWeights", x => uOutputWeights[x.Layer]).AddGradient("DUOutputWeights", x => uOutputWeightsGradient[x.Layer])
                .AddWeight("WOutputVectors", x => wOutputVectors[x.Layer]).AddGradient("DWOutputVectors", x => wOutputVectorsGradient[x.Layer])
                .AddWeight("UOutputVectors", x => uOutputVectors[x.Layer]).AddGradient("DUOutputVectors", x => uOutputVectorsGradient[x.Layer])
                .AddWeight("OKeys", x => oKeys[x.Layer]).AddGradient("DOKeys", x => oKeysGradient[x.Layer])
                .AddWeight("OKB", x => oKB[x.Layer]).AddGradient("DOKB", x => oKBGradient[x.Layer])
                .AddWeight("HKeys", x => hKeys[x.Layer]).AddGradient("DHKeys", x => hKeysGradient[x.Layer])
                .AddWeight("HKB", x => hKB[x.Layer]).AddGradient("DHKB", x => hKBGradient[x.Layer])
                .AddWeight("PreviousWeights", x => previousWeights[x.Layer]).AddGradient("DPreviousWeights", x => previousWeightsGradient[x.Layer])
                .AddWeight("CWeights", x => cWeights[x.Layer]).AddGradient("DCWeights", x => cWeightsGradient[x.Layer])
                .AddWeight("HiddenWeights", x => hiddenWeights[x.Layer]).AddGradient("DHiddenWeights", x => hiddenWeightsGradient[x.Layer])
                .AddWeight("RowSumWeights", x => rowSumWeights).AddGradient("DRowSumWeights", x => rowSumWeightsGradient)
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
