//------------------------------------------------------------------------------
// <copyright file="SelfAttentionMultiLayerLSTM.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.LstmExample
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.LstmExample.RMAD;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A multi-layer LSTM with self-attention.
    /// </summary>
    public class SelfAttentionMultiLayerLSTM : NeuralNetwork, ILSTM
    {
        private readonly int hiddenSize;
        private readonly double clipValue;
        private readonly string architecture;
        private readonly int originalInputSize;
        private readonly int inputSize;
        private readonly int outputSize;
        private readonly int numTimeSteps;
        private readonly int numLayers;
        private readonly string lstmName;

        private Matrix[][] h;
        private Matrix[][] c; // Memory cell state
        private Matrix[] output;

        private IModelLayer embeddingLayer;
        private IModelLayer hiddenLayer;
        private IModelLayer outputLayer;

        private Matrix[][][] arrays4D;
        private Matrix[][] arrays3D;

        private SelfAttentionMultiLayerLSTMComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="SelfAttentionMultiLayerLSTM"/> class.
        /// </summary>
        /// <param name="inputSize">The input size.</param>
        /// <param name="hiddenSize">The hidden size.</param>
        /// <param name="outputSize">The output size.</param>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="architecture">The name of the JSON architecture.</param>
        /// <param name="lstmName">The name of the LSTM.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="clipValue">The clip value.</param>
        public SelfAttentionMultiLayerLSTM(int inputSize, int hiddenSize, int outputSize, int numTimeSteps, double learningRate, string architecture, string lstmName, int numLayers = 2, double clipValue = 4.0d)
        {
            this.inputSize = hiddenSize;
            this.originalInputSize = inputSize;
            inputSize = hiddenSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.Parameters.LearningRate = learningRate;
            this.numTimeSteps = numTimeSteps;
            this.numLayers = numLayers;
            this.clipValue = clipValue;
            this.Parameters.NumTimeSteps = numTimeSteps;
            this.architecture = architecture;
            this.lstmName = lstmName;

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

        /// <inheritdoc/>
        public string Name
        {
            get
            {
                return this.lstmName;
            }
        }

        /// <summary>
        /// Serialize LSTMParameters to a JSON string.
        /// </summary>
        /// <param name="parameters">The parameters to serialize.</param>
        /// <returns>The serialized parameters.</returns>
        public static string SerializeLSTMParameters(MultiLayerLSTMParameters parameters)
        {
            string jsonString = JsonConvert.SerializeObject(parameters, Newtonsoft.Json.Formatting.Indented);
            return jsonString;
        }

        /// <summary>
        /// Deserialize a JSON string back to LSTMParameters.
        /// </summary>
        /// <param name="jsonString">The JSON string to deserialize.</param>
        /// <returns>The multi-layer LSTM parameters.</returns>
        public static MultiLayerLSTMParameters DeserializeLSTMParameters(string jsonString)
        {
            MultiLayerLSTMParameters parameters = JsonConvert.DeserializeObject<MultiLayerLSTMParameters>(jsonString) ?? throw new InvalidOperationException("There was a problem deserializing the JSON stirng.");

            return parameters;
        }

        /// <summary>
        /// Initializes the computation graph of the LSTM.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            await this.InitializeComputationGraph();
        }

        /// <inheritdoc/>
        public double[] GetOutput(Matrix[] inputs)
        {
            // Initialize memory cell, hidden state, biases, and intermediates
            this.ClearState();

            // Forward propagate through the MultiLayerLSTM
            MatrixUtils.SetInPlace(this.Parameters.InputSequence, inputs);
            var op = this.computationGraph.StartOperation ?? throw new Exception("Start operation should not be null.");
            IOperationBase? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);
                var forwardMethod = op.OperationType.GetMethod("Forward") ?? throw new Exception($"Forward method should exist on operation of type {op.OperationType.Name}.");
                forwardMethod.Invoke(op, parameters);
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

            return MatrixUtils.To1DArray(this.output);
        }

        /// <inheritdoc/>
        public async Task Optimize(Matrix[] inputs, List<Matrix> chosenActions, List<double> rewards, int iterationIndex, bool doNotUpdate = false)
        {
            this.Parameters.AdamIteration = iterationIndex + 1;

            await this.AutomaticForwardPropagate(inputs, chosenActions, rewards, doNotUpdate);
        }

        /// <inheritdoc/>
        public void SaveModel(string filePath)
        {
            MultiLayerLSTMParameters parameters = new MultiLayerLSTMParameters
            {
                Wi = this.hiddenLayer.WeightDeepMatrix("Wi"),
                Wf = this.hiddenLayer.WeightDeepMatrix("Wf"),
                Wc = this.hiddenLayer.WeightDeepMatrix("Wc"),
                Wo = this.hiddenLayer.WeightDeepMatrix("Wo"),
                Ui = this.hiddenLayer.WeightDeepMatrix("Ui"),
                Uf = this.hiddenLayer.WeightDeepMatrix("Uf"),
                Uc = this.hiddenLayer.WeightDeepMatrix("Uc"),
                Uo = this.hiddenLayer.WeightDeepMatrix("Uo"),
                Bi = this.hiddenLayer.WeightDeepMatrix("bi"),
                Bf = this.hiddenLayer.WeightDeepMatrix("bf"),
                Bc = this.hiddenLayer.WeightDeepMatrix("bc"),
                Bo = this.hiddenLayer.WeightDeepMatrix("bo"),
                Be = this.embeddingLayer.WeightMatrix("be"),
                We = this.embeddingLayer.WeightMatrix("We"),
                V = this.outputLayer.WeightMatrix("V"),
                B = this.outputLayer.WeightMatrix("b"),
                Wq = this.hiddenLayer.WeightDeepMatrix("Wq"),
                Wk = this.hiddenLayer.WeightDeepMatrix("Wk"),
                Wv = this.hiddenLayer.WeightDeepMatrix("Wv"),
            };

            var serialized = SerializeLSTMParameters(parameters);
            File.WriteAllText(filePath, serialized);
        }

        private void InitializeState()
        {
            // Clear the hidden state and memory cell state
            this.h = new Matrix[this.Parameters.NumTimeSteps][];
            this.c = new Matrix[this.Parameters.NumTimeSteps][];
            for (int t = 0; t < this.Parameters.NumTimeSteps; ++t)
            {
                this.h[t] = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, 1);
                this.c[t] = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, 1);
            }

            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });

            // Clear intermediates
            this.output = MatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.outputSize, 1);
            this.Parameters.InputSequence = MatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.originalInputSize, 1);

            this.arrays4D = new Matrix[][][] { this.h, this.c };
            this.arrays3D = new Matrix[][] { this.output, this.Parameters.InputSequence };
        }

        private void ClearState()
        {
            MatrixUtils.ClearArrays4D(this.arrays4D);
            MatrixUtils.ClearArrays3D(this.arrays3D);
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

            string json = EmbeddedResource.ReadAllJson(this.architecture);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new SelfAttentionMultiLayerLSTMComputationGraph(this);
            var zeroMatrixHiddenSize = new Matrix(this.hiddenSize, 1);
            this.computationGraph
                .AddIntermediate("inputSequence", x => this.Parameters.InputSequence[x.TimeStep])
                .AddIntermediate("output", x => this.output[x.TimeStep])
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
                .AddOperationFinder("hFromCurrentTimeStepAndLastLayer", x => this.computationGraph[$"h_{x.TimeStep}_{this.numLayers - 1}"])
                .AddOperationFinder("currentInput", x => x.Layer == 0 ? this.computationGraph[$"embeddedInput_{x.TimeStep}_0"] : this.computationGraph[$"h_{x.TimeStep}_{x.Layer - 1}"])
                .AddOperationFinder("previousHiddenState", x => x.TimeStep == 0 ? zeroMatrixHiddenSize : this.computationGraph[$"h_{x.TimeStep - 1}_{x.Layer}"])
                .AddOperationFinder("previousMemoryCellState", x => x.TimeStep == 0 ? zeroMatrixHiddenSize : this.computationGraph[$"c_{x.TimeStep - 1}_{x.Layer}"])
                .ConstructFromArchitecture(jsonArchitecture, this.numTimeSteps, this.numLayers);

            IOperationBase? backwardStartOperation = null;
            for (int t = this.Parameters.NumTimeSteps - 1; t >= 0; t--)
            {
                backwardStartOperation = this.computationGraph[$"output_t_{t}_0"];
                OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                await opVisitor.TraverseAsync();
                await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
            }
        }

        private async Task AutomaticForwardPropagate(Matrix[] inputSequence, List<Matrix> chosenActions, List<double> rewards, bool doNotUpdate = false)
        {
            // Initialize memory cell, hidden state, gradients, biases, and intermediates
            this.ClearState();

            MatrixUtils.SetInPlace(this.Parameters.InputSequence, inputSequence);
            this.Parameters.ChosenActions = chosenActions;
            this.Parameters.Rewards = rewards;
            var op = this.computationGraph.StartOperation ?? throw new Exception("Start operation should not be null.");
            IOperationBase? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);
                var forwardMethod = op.OperationType.GetMethod("Forward") ?? throw new Exception($"Forward method should exist on operation of type {op.OperationType.Name}.");
                forwardMethod.Invoke(op, parameters);
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

        private async Task AutomaticBackwardPropagate(bool doNotUpdate = false)
        {
            var lossFunction = PolicyGradientLossOperation.Instantiate(this);
            var policyGradientLossOperation = (PolicyGradientLossOperation)lossFunction;
            var loss = policyGradientLossOperation.Forward(this.output);
            if (loss[0][0] >= 0.0d)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }

            Console.WriteLine($"{this.lstmName}: Policy gradient loss: {loss[0][0]}");
            Console.ForegroundColor = ConsoleColor.White;
            var gradientOfLossWrtOutput = lossFunction.Backward(MatrixUtils.To2DArray(this.output)).Item1 as Matrix ?? throw new Exception("Gradient of the loss wrt the output should not be null.");
            int traverseCount = 0;
            IOperationBase? backwardStartOperation = null;
            for (int t = this.Parameters.NumTimeSteps - 1; t >= 0; t--)
            {
                backwardStartOperation = this.computationGraph[$"output_t_{t}_0"];
                if (gradientOfLossWrtOutput[t][0] != 0.0d)
                {
                    var backwardInput = new Matrix(1, 1);
                    backwardInput[0] = gradientOfLossWrtOutput[t];
                    backwardStartOperation.BackwardInput = backwardInput;
                    OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                    await opVisitor.TraverseAsync();
                    opVisitor.Reset();
                    traverseCount++;
                }
            }

            if (traverseCount == 0 || doNotUpdate)
            {
                return;
            }

            GradientClipper clipper = new GradientClipper(this);
            clipper.Clip(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });

            AdamOptimizer optimizer = new AdamOptimizer(this);
            optimizer.Optimize(new[] { this.embeddingLayer, this.hiddenLayer, this.outputLayer });
        }
    }
}