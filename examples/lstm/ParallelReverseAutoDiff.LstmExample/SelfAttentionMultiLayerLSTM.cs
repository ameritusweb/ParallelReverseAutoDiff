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

        private Matrix V;
        private Matrix dV;

        private Matrix b;
        private Matrix db;

        private Matrix[] bi;
        private Matrix[] dbi;

        private Matrix[] bf;
        private Matrix[] dbf;

        private Matrix[] Wi;
        private Matrix[] dWi;

        private Matrix[] Wf;
        private Matrix[] dWf;

        private Matrix[] Wo;
        private Matrix[] dWo;

        private Matrix[] Uo;
        private Matrix[] dUo;

        private Matrix[] bo;
        private Matrix[] dbo;

        private Matrix[] Ui;
        private Matrix[] dUi;

        private Matrix[] Uf;
        private Matrix[] dUf;

        private Matrix[] Wc;
        private Matrix[] dWc;

        private Matrix[] Uc;
        private Matrix[] dUc;

        private Matrix[] bc;
        private Matrix[] dbc;

        private Matrix[] Wq;
        private Matrix[] dWq;

        private Matrix[] Wk;
        private Matrix[] dWk;

        private Matrix[] Wv;
        private Matrix[] dWv;

        private Matrix We;
        private Matrix be;

        private Matrix dWe;
        private Matrix dbe;

        private Matrix mV;
        private Matrix vV;
        private Matrix mb;
        private Matrix vb;

        private Matrix[] mWi;
        private Matrix[] vWi;
        private Matrix[] mWf;
        private Matrix[] vWf;
        private Matrix[] mWc;
        private Matrix[] vWc;
        private Matrix[] mWo;
        private Matrix[] vWo;

        private Matrix[] mUi;
        private Matrix[] vUi;
        private Matrix[] mUf;
        private Matrix[] vUf;
        private Matrix[] mUc;
        private Matrix[] vUc;
        private Matrix[] mUo;
        private Matrix[] vUo;

        private Matrix[] mbi;
        private Matrix[] vbi;
        private Matrix[] mbf;
        private Matrix[] vbf;
        private Matrix[] mbc;
        private Matrix[] vbc;
        private Matrix[] mbo;
        private Matrix[] vbo;
        private Matrix mWe;
        private Matrix vWe;
        private Matrix mbe;
        private Matrix vbe;

        private Matrix[] mWq;
        private Matrix[] vWq;
        private Matrix[] mWk;
        private Matrix[] vWk;
        private Matrix[] mWv;
        private Matrix[] vWv;

        private int adamT;

        private Matrix[][][] arrays4D;
        private Matrix[][] arrays3D;
        private Matrix[] arrays2D;

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

            // Initialize model parameters
            this.Wo = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            this.Uo = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            this.bo = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            this.Wi = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            this.Ui = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            this.bi = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            this.Wf = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            this.Uf = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            this.bf = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, outputSize);

            this.Wc = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            this.Uc = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            this.bc = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            this.Wq = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            this.Wk = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            this.Wv = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);

            this.We = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(hiddenSize, this.originalInputSize);
            this.be = MatrixUtils.InitializeZeroMatrix(hiddenSize, outputSize);

            this.V = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(outputSize, hiddenSize);
            this.b = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(outputSize, 1);

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
        /// Load the model from a JSON file.
        /// </summary>
        /// <param name="filePath">The file path to load from.</param>
        /// <param name="lstm">The LSTM to load the parameters to.</param>
        /// <returns>The loaded LSTM.</returns>
        public static SelfAttentionMultiLayerLSTM LoadModel(string filePath, SelfAttentionMultiLayerLSTM lstm)
        {
            string jsonString = File.ReadAllText(filePath);
            var parameters = DeserializeLSTMParameters(jsonString);

            lstm.Wi = parameters.Wi;
            lstm.Wf = parameters.Wf;
            lstm.Wc = parameters.Wc;
            lstm.Wo = parameters.Wo;
            lstm.Ui = parameters.Ui;
            lstm.Uf = parameters.Uf;
            lstm.Uc = parameters.Uc;
            lstm.Uo = parameters.Uo;
            lstm.bi = parameters.Bi;
            lstm.bf = parameters.Bf;
            lstm.bc = parameters.Bc;
            lstm.bo = parameters.Bo;
            lstm.be = parameters.Be;
            lstm.We = parameters.We;
            lstm.Wq = parameters.Wq;
            lstm.Wk = parameters.Wk;
            lstm.Wv = parameters.Wv;
            lstm.V = parameters.V;
            lstm.b = parameters.B;

            return lstm;
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
            IOperation? currOp = null;
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
            this.adamT = iterationIndex + 1;

            await this.AutomaticForwardPropagate(inputs, chosenActions, rewards, doNotUpdate);
        }

        /// <inheritdoc/>
        public void SaveModel(string filePath)
        {
            MultiLayerLSTMParameters parameters = new MultiLayerLSTMParameters
            {
                Wi = this.Wi,
                Wf = this.Wf,
                Wc = this.Wc,
                Wo = this.Wo,
                Ui = this.Ui,
                Uf = this.Uf,
                Uc = this.Uc,
                Uo = this.Uo,
                Bi = this.bi,
                Bf = this.bf,
                Bc = this.bc,
                Bo = this.bo,
                Be = this.be,
                We = this.We,
                V = this.V,
                B = this.b,
                Wq = this.Wq,
                Wk = this.Wk,
                Wv = this.Wv,
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

            // Clear gradients and biases
            this.dWo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.inputSize);
            this.dUo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.hiddenSize);
            this.dbo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.outputSize);

            this.dWi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.inputSize);
            this.dUi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.hiddenSize);
            this.dbi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.outputSize);

            this.dWf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.inputSize);
            this.dUf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.hiddenSize);
            this.dbf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.outputSize);

            this.dWc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.inputSize);
            this.dUc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.hiddenSize);
            this.dbc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.outputSize);

            this.dWq = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.hiddenSize);
            this.dWk = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.hiddenSize);
            this.dWv = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.hiddenSize, this.hiddenSize);

            this.dWe = MatrixUtils.InitializeZeroMatrix(this.hiddenSize, this.originalInputSize);
            this.dbe = MatrixUtils.InitializeZeroMatrix(this.hiddenSize, this.outputSize);

            this.dV = MatrixUtils.InitializeZeroMatrix(this.outputSize, this.hiddenSize);
            this.db = MatrixUtils.InitializeZeroMatrix(this.outputSize, 1);

            // Clear intermediates
            this.output = MatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.outputSize, 1);
            this.Parameters.InputSequence = MatrixUtils.InitializeZeroMatrix(this.Parameters.NumTimeSteps, this.originalInputSize, 1);

            this.arrays4D = new Matrix[][][] { this.h, this.c };
            this.arrays3D = new Matrix[][] { this.dWo, this.dUo, this.dbo, this.dWi, this.dUi, this.dbi, this.dWf, this.dUf, this.dbf, this.dWc, this.dUc, this.dbc, this.dWq, this.dWk, this.dWv, this.output, this.Parameters.InputSequence };
            this.arrays2D = new[] { this.dWe, this.dbe, this.dV, this.db };
        }

        private void ClearState()
        {
            MatrixUtils.ClearArrays4D(this.arrays4D);
            MatrixUtils.ClearArrays3D(this.arrays3D);
            MatrixUtils.ClearArrays2D(this.arrays2D);
        }

        private async Task InitializeComputationGraph()
        {
            string json = EmbeddedResource.ReadAllJson(this.architecture);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new SelfAttentionMultiLayerLSTMComputationGraph(this);
            var zeroMatrixHiddenSize = MatrixUtils.InitializeZeroMatrix(this.hiddenSize, 1);
            this.computationGraph
                .AddIntermediate("inputSequence", x => this.Parameters.InputSequence[x.TimeStep])
                .AddIntermediate("output", x => this.output[x.TimeStep])
                .AddIntermediate("c", x => this.c[x.TimeStep][x.Layer])
                .AddIntermediate("h", x => this.h[x.TimeStep][x.Layer])
                .AddScalar("scaledDotProductScalar", x => 1.0d / Math.Sqrt(this.hiddenSize))
                .AddWeight("Wf", x => this.Wf[x.Layer]).AddGradient("dWf", x => this.dWf[x.Layer])
                .AddWeight("Wi", x => this.Wi[x.Layer]).AddGradient("dWi", x => this.dWi[x.Layer])
                .AddWeight("Wc", x => this.Wc[x.Layer]).AddGradient("dWc", x => this.dWc[x.Layer])
                .AddWeight("Wo", x => this.Wo[x.Layer]).AddGradient("dWo", x => this.dWo[x.Layer])
                .AddWeight("Uf", x => this.Uf[x.Layer]).AddGradient("dUf", x => this.dUf[x.Layer])
                .AddWeight("Ui", x => this.Ui[x.Layer]).AddGradient("dUi", x => this.dUi[x.Layer])
                .AddWeight("Uc", x => this.Uc[x.Layer]).AddGradient("dUc", x => this.dUc[x.Layer])
                .AddWeight("Uo", x => this.Uo[x.Layer]).AddGradient("dUo", x => this.dUo[x.Layer])
                .AddWeight("bf", x => this.bf[x.Layer]).AddGradient("dbf", x => this.dbf[x.Layer])
                .AddWeight("bi", x => this.bi[x.Layer]).AddGradient("dbi", x => this.dbi[x.Layer])
                .AddWeight("bc", x => this.bc[x.Layer]).AddGradient("dbc", x => this.dbc[x.Layer])
                .AddWeight("bo", x => this.bo[x.Layer]).AddGradient("dbo", x => this.dbo[x.Layer])
                .AddWeight("Wq", x => this.Wq[x.Layer]).AddGradient("dWq", x => this.dWq[x.Layer])
                .AddWeight("Wk", x => this.Wk[x.Layer]).AddGradient("dWk", x => this.dWk[x.Layer])
                .AddWeight("Wv", x => this.Wv[x.Layer]).AddGradient("dWv", x => this.dWv[x.Layer])
                .AddWeight("We", x => this.We).AddGradient("dWe", x => this.dWe)
                .AddWeight("be", x => this.be).AddGradient("dbe", x => this.dbe)
                .AddWeight("V", x => this.V).AddGradient("dV", x => this.dV)
                .AddWeight("b", x => this.b).AddGradient("db", x => this.db)
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

            IOperation? backwardStartOperation = null;
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
            IOperation? currOp = null;
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
            var gradientOfLossWrtOutput = lossFunction.Backward(MatrixUtils.To2DArray(this.output)).Item1 ?? throw new Exception("Gradient of the loss wrt the output should not be null.");
            int traverseCount = 0;
            IOperation? backwardStartOperation = null;
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

            // Clip gradients and biases to prevent exploding gradients
            this.dWi = MatrixUtils.ClipGradients(this.dWi, this.clipValue);
            this.dWf = MatrixUtils.ClipGradients(this.dWf, this.clipValue);
            this.dWo = MatrixUtils.ClipGradients(this.dWo, this.clipValue);
            this.dWc = MatrixUtils.ClipGradients(this.dWc, this.clipValue);
            this.dUi = MatrixUtils.ClipGradients(this.dUi, this.clipValue);
            this.dUf = MatrixUtils.ClipGradients(this.dUf, this.clipValue);
            this.dUo = MatrixUtils.ClipGradients(this.dUo, this.clipValue);
            this.dUc = MatrixUtils.ClipGradients(this.dUc, this.clipValue);
            this.dbi = MatrixUtils.ClipGradients(this.dbi, this.clipValue);
            this.dbf = MatrixUtils.ClipGradients(this.dbf, this.clipValue);
            this.dbo = MatrixUtils.ClipGradients(this.dbo, this.clipValue);
            this.dbc = MatrixUtils.ClipGradients(this.dbc, this.clipValue);
            this.dV = MatrixUtils.ClipGradients(this.dV, this.clipValue);
            this.db = MatrixUtils.ClipGradients(this.db, this.clipValue);
            this.dWe = MatrixUtils.ClipGradients(this.dWe, this.clipValue);
            this.dbe = MatrixUtils.ClipGradients(this.dbe, this.clipValue);
            this.dWq = MatrixUtils.ClipGradients(this.dWq, this.clipValue);
            this.dWk = MatrixUtils.ClipGradients(this.dWk, this.clipValue);
            this.dWv = MatrixUtils.ClipGradients(this.dWv, this.clipValue);

            // Update model parameters using gradient descent
            this.UpdateParametersWithAdam(this.dWi, this.dWf, this.dWo, this.dWc, this.dUi, this.dUf, this.dUo, this.dUc, this.dbi, this.dbf, this.dbo, this.dbc, this.dV, this.db, this.dWq, this.dWk, this.dWv, this.dWe, this.dbe);
        }

        private void UpdateParametersWithAdam(Matrix[] dWi, Matrix[] dWf, Matrix[] dWo, Matrix[] dWc, Matrix[] dUi, Matrix[] dUf, Matrix[] dUo, Matrix[] dUc, Matrix[] dbi, Matrix[] dbf, Matrix[] dbo, Matrix[] dbc, Matrix dV, Matrix db, Matrix[] dWq, Matrix[] dWk, Matrix[] dWv, Matrix dWe, Matrix dbe)
        {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            // Initialize moments
            if (this.mWi == null)
            {
                this.mWi = this.InitializeMoments() ?? throw new Exception("Moments should not be null.");
            }

            // Use Parallel.For to parallelize the loop
            Parallel.For(0, this.numLayers, layerIndex =>
            {
                // Update moments and apply Adam updates
                this.UpdateWeightWithAdam(this.Wi[layerIndex], this.mWi[layerIndex], this.vWi[layerIndex], dWi[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wf[layerIndex], this.mWf[layerIndex], this.vWf[layerIndex], dWf[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wc[layerIndex], this.mWc[layerIndex], this.vWc[layerIndex], dWc[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wo[layerIndex], this.mWo[layerIndex], this.vWo[layerIndex], dWo[layerIndex], beta1, beta2, epsilon, this.adamT);

                this.UpdateWeightWithAdam(this.Ui[layerIndex], this.mUi[layerIndex], this.vUi[layerIndex], dUi[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Uf[layerIndex], this.mUf[layerIndex], this.vUf[layerIndex], dUf[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Uc[layerIndex], this.mUc[layerIndex], this.vUc[layerIndex], dUc[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Uo[layerIndex], this.mUo[layerIndex], this.vUo[layerIndex], dUo[layerIndex], beta1, beta2, epsilon, this.adamT);

                this.UpdateWeightWithAdam(this.bi[layerIndex], this.mbi[layerIndex], this.vbi[layerIndex], dbi[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.bf[layerIndex], this.mbf[layerIndex], this.vbf[layerIndex], dbf[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.bc[layerIndex], this.mbc[layerIndex], this.vbc[layerIndex], dbc[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.bo[layerIndex], this.mbo[layerIndex], this.vbo[layerIndex], dbo[layerIndex], beta1, beta2, epsilon, this.adamT);

                this.UpdateWeightWithAdam(this.Wq[layerIndex], this.mWq[layerIndex], this.vWq[layerIndex], dWq[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wk[layerIndex], this.mWk[layerIndex], this.vWk[layerIndex], dWk[layerIndex], beta1, beta2, epsilon, this.adamT);
                this.UpdateWeightWithAdam(this.Wv[layerIndex], this.mWv[layerIndex], this.vWv[layerIndex], dWv[layerIndex], beta1, beta2, epsilon, this.adamT);
            });

            var frobeniusNorm = MatrixUtils.FrobeniusNorm(this.V);
            var learningRateReductionFactor = MatrixUtils.LearningRateReductionFactor(frobeniusNorm, 1.4d, 0.001d);
            Console.WriteLine($"{this.lstmName}: Frobenius norm: {frobeniusNorm}, learning rate reduction factor: {learningRateReductionFactor}");
            this.UpdateWeightWithAdam(this.V, this.mV, this.vV, dV, beta1, beta2, epsilon, this.adamT, this.Parameters.LearningRate * learningRateReductionFactor);
            this.UpdateWeightWithAdam(this.b, this.mb, this.vb, db, beta1, beta2, epsilon, this.adamT, this.Parameters.LearningRate * learningRateReductionFactor);

            this.UpdateWeightWithAdam(this.We, this.mWe, this.vWe, dWe, beta1, beta2, epsilon, this.adamT);
            this.UpdateWeightWithAdam(this.be, this.mbe, this.vbe, dbe, beta1, beta2, epsilon, this.adamT);
        }

        private Matrix[] InitializeMoments()
        {
            this.mWi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wi[0].Length, this.Wi[0][0].Length);
            this.vWi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wi[0].Length, this.Wi[0][0].Length);
            this.mWf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wf[0].Length, this.Wf[0][0].Length);
            this.vWf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wf[0].Length, this.Wf[0][0].Length);
            this.mWc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wc[0].Length, this.Wc[0][0].Length);
            this.vWc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wc[0].Length, this.Wc[0][0].Length);
            this.mWo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wo[0].Length, this.Wo[0][0].Length);
            this.vWo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wo[0].Length, this.Wo[0][0].Length);

            this.mWq = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wq[0].Length, this.Wq[0][0].Length);
            this.vWq = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wq[0].Length, this.Wq[0][0].Length);
            this.mWk = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wk[0].Length, this.Wk[0][0].Length);
            this.vWk = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wk[0].Length, this.Wk[0][0].Length);
            this.mWv = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wv[0].Length, this.Wv[0][0].Length);
            this.vWv = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Wv[0].Length, this.Wv[0][0].Length);

            this.mUi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Ui[0].Length, this.Ui[0][0].Length);
            this.vUi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Ui[0].Length, this.Ui[0][0].Length);
            this.mUf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Uf[0].Length, this.Uf[0][0].Length);
            this.vUf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Uf[0].Length, this.Uf[0][0].Length);
            this.mUc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Uc[0].Length, this.Uc[0][0].Length);
            this.vUc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Uc[0].Length, this.Uc[0][0].Length);
            this.mUo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Uo[0].Length, this.Uo[0][0].Length);
            this.vUo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.Uo[0].Length, this.Uo[0][0].Length);

            this.mbi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bi[0].Length, this.bi[0][0].Length);
            this.vbi = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bi[0].Length, this.bi[0][0].Length);
            this.mbf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bf[0].Length, this.bf[0][0].Length);
            this.vbf = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bf[0].Length, this.bf[0][0].Length);
            this.mbc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bc[0].Length, this.bc[0][0].Length);
            this.vbc = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bc[0].Length, this.bc[0][0].Length);
            this.mbo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bo[0].Length, this.bo[0][0].Length);
            this.vbo = MatrixUtils.InitializeZeroMatrix(this.numLayers, this.bo[0].Length, this.bo[0][0].Length);
            this.mWe = MatrixUtils.InitializeZeroMatrix(this.We.Length, this.We[0].Length);
            this.vWe = MatrixUtils.InitializeZeroMatrix(this.We.Length, this.We[0].Length);
            this.mbe = MatrixUtils.InitializeZeroMatrix(this.be.Length, this.be[0].Length);
            this.vbe = MatrixUtils.InitializeZeroMatrix(this.be.Length, this.be[0].Length);
            this.mV = MatrixUtils.InitializeZeroMatrix(this.V.Length, this.V[0].Length);
            this.vV = MatrixUtils.InitializeZeroMatrix(this.V.Length, this.V[0].Length);
            this.mb = MatrixUtils.InitializeZeroMatrix(this.b.Length, this.b[0].Length);
            this.vb = MatrixUtils.InitializeZeroMatrix(this.b.Length, this.b[0].Length);
            return this.mWi;
        }

        private void UpdateWeightWithAdam(Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon, int t, double? newLearningRate = null)
        {
            var lr = newLearningRate.HasValue ? newLearningRate.Value : this.Parameters.LearningRate;

            // Update biased first moment estimate
            mW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta1, mW), MatrixUtils.ScalarMultiply(1 - beta1, gradient));

            // Update biased second raw moment estimate
            vW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta2, vW), MatrixUtils.ScalarMultiply(1 - beta2, MatrixUtils.HadamardProduct(gradient, gradient)));

            // Compute bias-corrected first moment estimate
            Matrix mW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, t)), mW);

            // Compute bias-corrected second raw moment estimate
            Matrix vW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, t)), vW);

            // Update weights
            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    double weightReductionValue = lr * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);
                    w[i][j] -= weightReductionValue;
                }
            }
        }
    }
}