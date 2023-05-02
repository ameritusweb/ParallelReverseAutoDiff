namespace ParallelReverseAutoDiff.LstmExample
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.LstmExample.RMAD;
    using ParallelReverseAutoDiff.RMAD;

    public class SelfAttentionMultiLayerLSTM : NeuralNetwork, ILSTM
    {
        private double[][][][] h;
        private double[][][][] c; // Memory cell state
        private double[][][][] i;
        private double[][][][] f;
        private double[][][][] cHat;
        private double[][][][] o;
        private double[][][] output;

        private int originalInputSize;
        private int inputSize;
        private int hiddenSize;

        private int outputSize;

        private double[][] V;
        private double[][] dV;

        private double[][] b;
        private double[][] db;

        private double[][][] bi;
        private double[][][] dbi;

        private double[][][] bf;
        private double[][][] dbf;

        private double[][][] Wi;
        private double[][][] dWi;

        private double[][][] Wf;
        private double[][][] dWf;

        private double[][][] Wo;
        private double[][][] dWo;

        private double[][][] Uo;
        private double[][][] dUo;

        private double[][][] bo;
        private double[][][] dbo;

        private double[][][] Ui;
        private double[][][] dUi;

        private double[][][] Uf;
        private double[][][] dUf;

        private double[][][] Wc;
        private double[][][] dWc;

        private double[][][] Uc;
        private double[][][] dUc;

        private double[][][] bc;
        private double[][][] dbc;

        private double[][][] Wq;
        private double[][][] dWq;

        private double[][][] Wk;
        private double[][][] dWk;

        private double[][][] Wv;
        private double[][][] dWv;

        private double[][] We;
        private double[][] be;

        private double[][] dWe;
        private double[][] dbe;

        private double[][] mV;
        private double[][] vV;
        private double[][] mb;
        private double[][] vb;

        private double[][][] mWi;
        private double[][][] vWi;
        private double[][][] mWf;
        private double[][][] vWf;
        private double[][][] mWc;
        private double[][][] vWc;
        private double[][][] mWo;
        private double[][][] vWo;

        private double[][][] mUi;
        private double[][][] vUi;
        private double[][][] mUf;
        private double[][][] vUf;
        private double[][][] mUc;
        private double[][][] vUc;
        private double[][][] mUo;
        private double[][][] vUo;

        private double[][][] mbi;
        private double[][][] vbi;
        private double[][][] mbf;
        private double[][][] vbf;
        private double[][][] mbc;
        private double[][][] vbc;
        private double[][][] mbo;
        private double[][][] vbo;
        private double[][] mWe;
        private double[][] vWe;
        private double[][] mbe;
        private double[][] vbe;

        private double[][][] mWq;
        private double[][][] vWq;
        private double[][][] mWk;
        private double[][][] vWk;
        private double[][][] mWv;
        private double[][][] vWv;

        private Random rng;

        private double learningRate;
        private double clipValue;
        private int numLayers;
        private int adamT;

        private const string NAMESPACE = "ParallelReverseAutoDiff.LstmExample.architecture";
        private Dictionary<string, IOperation> operationsMap;
        private IOperation priorOperation;
        private IOperation startOperation;
        private IOperation backwardStartOperation;
        private Dictionary<string, Func<int, int, object>> _inputNameToValueMap;
        private double[][][][][] arrays4D;
        private double[][][][] arrays3D;
        private double[][][] arrays2D;
        private string architecture;
        private string lstmName;

        public SelfAttentionMultiLayerLSTM(int inputSize, int hiddenSize, int outputSize, int numTimeSteps, double learningRate, string architecture, string lstmName, int numLayers = 2, double clipValue = 4.0d)
        {
            this.inputSize = hiddenSize;
            this.originalInputSize = inputSize;
            inputSize = hiddenSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            rng = new Random(DateTime.UtcNow.Millisecond);
            this.learningRate = learningRate;
            this.numLayers = numLayers;
            this.clipValue = clipValue;
            this.numTimeSteps = numTimeSteps;
            this.architecture = architecture;
            this.lstmName = lstmName;

            // Initialize model parameters
            Wo = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            Uo = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            bo = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            Wi = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            Ui = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            bi = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            Wf = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            Uf = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            bf = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, outputSize);

            Wc = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, inputSize);
            Uc = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            bc = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            Wq = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            Wk = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);
            Wv = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(numLayers, hiddenSize, hiddenSize);

            We = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(hiddenSize, originalInputSize);
            be = MatrixUtils.InitializeZeroMatrix(hiddenSize, outputSize);

            V = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(outputSize, hiddenSize);
            b = MatrixUtils.InitializeRandomMatrixWithXavierInitialization(outputSize, 1);

            InitializeState();

            SetupInputNameToValueMap();
        }

        public string Name
        {
            get
            {
                return lstmName;
            }
        }

        public async Task Initialize()
        {
            await InitializeComputationGraph();
        }

        private void InitializeState()
        {
            // Clear the hidden state and memory cell state
            h = new double[numTimeSteps][][][];
            c = new double[numTimeSteps][][][];
            for (int t = 0; t < numTimeSteps; ++t)
            {
                h[t] = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, 1);
                c[t] = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, 1);
            }

            // Clear gradients and biases
            dWo = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, inputSize);
            dUo = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, hiddenSize);
            dbo = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            dWi = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, inputSize);
            dUi = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, hiddenSize);
            dbi = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            dWf = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, inputSize);
            dUf = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, hiddenSize);
            dbf = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            dWc = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, inputSize);
            dUc = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, hiddenSize);
            dbc = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, outputSize);

            dWq = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, hiddenSize);
            dWk = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, hiddenSize);
            dWv = MatrixUtils.InitializeZeroMatrix(numLayers, hiddenSize, hiddenSize);

            dWe = MatrixUtils.InitializeZeroMatrix(hiddenSize, originalInputSize);
            dbe = MatrixUtils.InitializeZeroMatrix(hiddenSize, outputSize);

            dV = MatrixUtils.InitializeZeroMatrix(outputSize, hiddenSize);
            db = MatrixUtils.InitializeZeroMatrix(outputSize, 1);

            // Clear intermediates
            f = MatrixUtils.InitializeZeroMatrix(numTimeSteps, numLayers, hiddenSize, outputSize);
            i = MatrixUtils.InitializeZeroMatrix(numTimeSteps, numLayers, hiddenSize, outputSize);
            cHat = MatrixUtils.InitializeZeroMatrix(numTimeSteps, numLayers, hiddenSize, outputSize);
            o = MatrixUtils.InitializeZeroMatrix(numTimeSteps, numLayers, hiddenSize, outputSize);
            output = MatrixUtils.InitializeZeroMatrix(numTimeSteps, outputSize, 1);
            inputSequence = MatrixUtils.InitializeZeroMatrix(numTimeSteps, originalInputSize, 1);

            arrays4D = new double[][][][][] { h, c, f, i, cHat, o };
            arrays3D = new double[][][][] { dWo, dUo, dbo, dWi, dUi, dbi, dWf, dUf, dbf, dWc, dUc, dbc, dWq, dWk, dWv, output, inputSequence };
            arrays2D = new double[][][] { dWe, dbe, dV, db };
        }

        private void ClearState()
        {
            MatrixUtils.ClearArrays4D(arrays4D);
            MatrixUtils.ClearArrays3D(arrays3D);
            MatrixUtils.ClearArrays2D(arrays2D);
        }

        private void SetupInputNameToValueMap()
        {
            var zeroMatrixHiddenSize = MatrixUtils.InitializeZeroMatrix(hiddenSize, 1);
            _inputNameToValueMap = new Dictionary<string, Func<int, int, object>>
            {
                { "inputSequence", (t, l) => inputSequence[t] },
                { "output", (t, l) => output[t] },
                { "Wf", (t, l) => Wf[l] },
                { "Wi", (t, l) => Wi[l] },
                { "Wc", (t, l) => Wc[l] },
                { "Wo", (t, l) => Wo[l] },
                { "Uf", (t, l) => Uf[l] },
                { "Ui", (t, l) => Ui[l] },
                { "Uc", (t, l) => Uc[l] },
                { "Uo", (t, l) => Uo[l] },
                { "bf", (t, l) => bf[l] },
                { "bi", (t, l) => bi[l] },
                { "bc", (t, l) => bc[l] },
                { "bo", (t, l) => bo[l] },
                { "dWf", (t, l) => dWf[l] },
                { "dWi", (t, l) => dWi[l] },
                { "dWc", (t, l) => dWc[l] },
                { "dWo", (t, l) => dWo[l] },
                { "dUf", (t, l) => dUf[l] },
                { "dUi", (t, l) => dUi[l] },
                { "dUc", (t, l) => dUc[l] },
                { "dUo", (t, l) => dUo[l] },
                { "dbf", (t, l) => dbf[l] },
                { "dbi", (t, l) => dbi[l] },
                { "dbc", (t, l) => dbc[l] },
                { "dbo", (t, l) => dbo[l] },
                { "f", (t, l) => operationsMap["f_" + t + "_" + l] },
                { "i", (t, l) => operationsMap["i_" + t + "_" + l] },
                { "cHat", (t, l) => operationsMap["cHat_" + t + "_" + l] },
                { "h", (t, l) => h[t][l] },
                { "c", (t, l) => c[t][l] },
                { "o", (t, l) => operationsMap["o_" + t + "_" + l] },
                { "We", (t, l) => We },
                { "be", (t, l) => be },
                { "Wq", (t, l) => Wq[l] },
                { "Wk", (t, l) => Wk[l] },
                { "Wv", (t, l) => Wv[l] },
                { "V", (t, l) => V },
                { "b", (t, l) => b },
                { "dWe", (t, l) => dWe },
                { "dbe", (t, l) => dbe },
                { "dWq", (t, l) => dWq[l] },
                { "dWk", (t, l) => dWk[l] },
                { "dWv", (t, l) => dWv[l] },
                { "dV", (t, l) => dV },
                { "db", (t, l) => db },
                { "scaledDotProductScalar", (t, l) => 1.0d / Math.Sqrt(hiddenSize) },
                { "embeddedInput", (t, l) => operationsMap["embeddedInput_" + t] },
                { "hFromCurrentTimeStepAndLastLayer", (t, l) => operationsMap["h_" + t + "_" + (numLayers - 1)] },
                { "currentInput", (t, l) => l == 0 ? operationsMap["embeddedInput_" + t] : operationsMap["h_" + t + "_" + (l - 1)] },
                { "previousHiddenState", (t, l) => t == 0 ? zeroMatrixHiddenSize : operationsMap["h_" + (t - 1) + "_" + l] },
                { "previousMemoryCellState", (t, l) => t == 0 ? zeroMatrixHiddenSize : operationsMap["c_" + (t - 1) + "_" + l] },
                // Add other input names and their corresponding getters here
            };
        }

        private string ConvertIdToTimeAndLayer(string id, string[] split, IOperation op)
        {
            if (split.Length == 1)
            {
                if (op.LayerIndex == -1)
                {
                    return id + "_" + op.TimeStepIndex;
                } else
                {
                    return id + "_" + op.TimeStepIndex + "_" + op.LayerIndex;
                }
            }
            return id;
        }

        private void SetupDependencies(IOperation op)
        {
            object[] parameters = new object[op.Inputs.Count];
            for (int j = 0; j < op.Inputs.Count; ++j)
            {
                var input = op.Inputs[j];
                var split = input.Split(new char[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                var specificId = ConvertIdToTimeAndLayer(input, split, op);
                if (operationsMap.ContainsKey(specificId))
                {
                    var inputOp = operationsMap[specificId];
                    inputOp.Outputs.Add(op.SpecificId);
                    op.BackwardAdjacentOperations.Add(inputOp);
                    parameters[j] = inputOp;
                    continue;
                }
                string inputName = split[0];

                if (_inputNameToValueMap.ContainsKey(inputName))
                {
                    int timeStepIndex = op.TimeStepIndex;
                    int layerIndex = op.LayerIndex;

                    // Get the corresponding value from the dictionary using the input name
                    var p = _inputNameToValueMap[inputName](timeStepIndex, layerIndex);
                    if (p is IOperation)
                    {
                        var inputOp = (IOperation) p;
                        inputOp.Outputs.Add(op.SpecificId);
                        op.BackwardAdjacentOperations.Add(inputOp);
                        parameters[j] = inputOp;
                        continue;
                    }
                    else
                    {
                        op.BackwardAdjacentOperations.Add(null);
                    }
                    parameters[j] = p;
                }
                else
                {
                    throw new Exception($"Input name {inputName} not found in value map");
                }

            }
            op.Parameters = parameters;
        }

        private object[] LookupParameters(IOperation op)
        {
            object[] parameters = op.Parameters;
            object[] parametersToReturn = new object[parameters.Length];
            for (int j = 0; j < parameters.Length; ++j)
            {
                if (parameters[j] is IOperation)
                {
                    parametersToReturn[j] = ((IOperation)parameters[j]).GetOutput();
                }
                else
                {
                    parametersToReturn[j] = parameters[j];
                }
            }
            return parametersToReturn;
        }

        private Func<int, int, object> NameToValueFunc(string name)
        {
            string[] split = name.Split(new char[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
            return _inputNameToValueMap[split[0]];
        }

        private async Task InitializeComputationGraph()
        {
            string json = EmbeddedResource.ReadAllJson(architecture);
            CreateOperationsFromJson(json);

            var op = startOperation;
            IOperation currOp = null;
            do
            {
                SetupDependencies(op);
                operationsMap[op.SpecificId] = op;
                currOp = op;
                if (op.HasNext)
                    op = op.Next;
            } while (currOp.Next != null);

            for (int t = numTimeSteps - 1; t >= 0; t--)
            {
                backwardStartOperation = operationsMap[$"output_t_{t}"];
                OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                await opVisitor.TraverseAsync();
                await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
            }
            Console.Clear();
        }

        private async Task AutomaticForwardPropagate(double[][][] inputSequence, List<double[][]> chosenActions, List<double> rewards, bool doNotUpdate = false)
        {
            // Initialize memory cell, hidden state, gradients, biases, and intermediates
            ClearState();

            MatrixUtils.SetInPlace(this.inputSequence, inputSequence);
            this.chosenActions = chosenActions;
            this.rewards = rewards;
            var op = startOperation;
            IOperation currOp = null;
            do
            {
                var parameters = LookupParameters(op);
                if (op.SpecificId.StartsWith("output"))
                {

                }
                op.OperationType.GetMethod("Forward").Invoke(op, parameters);
                if (op.ResultToName != null)
                {
                    op.ResultTo(NameToValueFunc(op.ResultToName));
                }
                operationsMap[op.SpecificId] = op;
                currOp = op;
                if (op.HasNext)
                    op = op.Next;
            } while (currOp.Next != null);

            await AutomaticBackwardPropagate(doNotUpdate);
        }

        private async Task AutomaticBackwardPropagate(bool doNotUpdate = false)
        {
            var lossFunction = PolicyGradientLossOperation.Instantiate(this);
            var policyGradientLossOperation = (PolicyGradientLossOperation)lossFunction;
            var loss = policyGradientLossOperation.Forward(output);
            if (loss[0][0] >= 0.0d)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            } else
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }
            Console.WriteLine($"{lstmName}: Policy gradient loss: {loss[0][0]}");
            Console.ForegroundColor = ConsoleColor.White;
            var gradientOfLossWrtOutput = lossFunction.Backward(MatrixUtils.To2DArray(output)).Item1;
            int traverseCount = 0;
            for (int t = numTimeSteps - 1; t >= 0; t--)
            {
                backwardStartOperation = operationsMap[$"output_t_{t}"];
                if (gradientOfLossWrtOutput[t][0] != 0.0d)
                {
                    backwardStartOperation.BackwardInput = new double[][] { gradientOfLossWrtOutput[t] };
                    OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, t);
                    await opVisitor.TraverseAsync();
                    opVisitor.Reset();
                    traverseCount++;
                }
            }

            if (traverseCount == 0.0d || doNotUpdate)
            {
                return;
            }

            // Clip gradients and biases to prevent exploding gradients
            dWi = MatrixUtils.ClipGradients(dWi, clipValue);
            dWf = MatrixUtils.ClipGradients(dWf, clipValue);
            dWo = MatrixUtils.ClipGradients(dWo, clipValue);
            dWc = MatrixUtils.ClipGradients(dWc, clipValue);
            dUi = MatrixUtils.ClipGradients(dUi, clipValue);
            dUf = MatrixUtils.ClipGradients(dUf, clipValue);
            dUo = MatrixUtils.ClipGradients(dUo, clipValue);
            dUc = MatrixUtils.ClipGradients(dUc, clipValue);
            dbi = MatrixUtils.ClipGradients(dbi, clipValue);
            dbf = MatrixUtils.ClipGradients(dbf, clipValue);
            dbo = MatrixUtils.ClipGradients(dbo, clipValue);
            dbc = MatrixUtils.ClipGradients(dbc, clipValue);
            dV = MatrixUtils.ClipGradients(dV, clipValue);
            db = MatrixUtils.ClipGradients(db, clipValue);
            dWe = MatrixUtils.ClipGradients(dWe, clipValue);
            dbe = MatrixUtils.ClipGradients(dbe, clipValue);
            dWq = MatrixUtils.ClipGradients(dWq, clipValue);
            dWk = MatrixUtils.ClipGradients(dWk, clipValue);
            dWv = MatrixUtils.ClipGradients(dWv, clipValue);

            // Update model parameters using gradient descent
            UpdateParametersWithAdam(dWi, dWf, dWo, dWc, dUi, dUf, dUo, dUc, dbi, dbf, dbo, dbc, dV, db, dWq, dWk, dWv, dWe, dbe);
        }

        public double[] GetOutput(double[][][] inputs)
        {
            // Initialize memory cell, hidden state, biases, and intermediates
            ClearState();

            // Forward propagate through the MultiLayerLSTM
            MatrixUtils.SetInPlace(this.inputSequence, inputs);
            var op = startOperation;
            IOperation currOp = null;
            do
            {
                var parameters = LookupParameters(op);
                if (op.SpecificId.StartsWith("output"))
                {

                }
                op.OperationType.GetMethod("Forward").Invoke(op, parameters);
                if (op.ResultToName != null)
                {
                    op.ResultTo(NameToValueFunc(op.ResultToName));
                }
                operationsMap[op.SpecificId] = op;
                currOp = op;
                if (op.HasNext)
                    op = op.Next;
            } while (currOp.Next != null);

            return MatrixUtils.To1DArray(output);
        }

        public async Task Optimize(double[][][] inputs, List<double[][]> chosenActions, List<double> rewards, int iterationIndex, bool doNotUpdate = false)
        {
            adamT = iterationIndex + 1;

            await AutomaticForwardPropagate(inputs, chosenActions, rewards, doNotUpdate);
        }

        private void UpdateParametersWithAdam(double[][][] dWi, double[][][] dWf, double[][][] dWo, double[][][] dWc, double[][][] dUi, double[][][] dUf, double[][][] dUo, double[][][] dUc, double[][][] dbi, double[][][] dbf, double[][][] dbo, double[][][] dbc, double[][] dV, double[][] db, double[][][] dWq, double[][][] dWk, double[][][] dWv, double[][] dWe, double[][] dbe)
        {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            // Initialize moments
            if (mWi == null)
            {
                InitializeMoments();
            }

            // Use Parallel.For to parallelize the loop
            Parallel.For(0, numLayers, layerIndex =>
            {
                // Update moments and apply Adam updates
                UpdateWeightWithAdam(Wi[layerIndex], mWi[layerIndex], vWi[layerIndex], dWi[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Wf[layerIndex], mWf[layerIndex], vWf[layerIndex], dWf[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Wc[layerIndex], mWc[layerIndex], vWc[layerIndex], dWc[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Wo[layerIndex], mWo[layerIndex], vWo[layerIndex], dWo[layerIndex], beta1, beta2, epsilon, adamT);

                UpdateWeightWithAdam(Ui[layerIndex], mUi[layerIndex], vUi[layerIndex], dUi[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Uf[layerIndex], mUf[layerIndex], vUf[layerIndex], dUf[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Uc[layerIndex], mUc[layerIndex], vUc[layerIndex], dUc[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Uo[layerIndex], mUo[layerIndex], vUo[layerIndex], dUo[layerIndex], beta1, beta2, epsilon, adamT);

                UpdateWeightWithAdam(bi[layerIndex], mbi[layerIndex], vbi[layerIndex], dbi[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(bf[layerIndex], mbf[layerIndex], vbf[layerIndex], dbf[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(bc[layerIndex], mbc[layerIndex], vbc[layerIndex], dbc[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(bo[layerIndex], mbo[layerIndex], vbo[layerIndex], dbo[layerIndex], beta1, beta2, epsilon, adamT);

                UpdateWeightWithAdam(Wq[layerIndex], mWq[layerIndex], vWq[layerIndex], dWq[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Wk[layerIndex], mWk[layerIndex], vWk[layerIndex], dWk[layerIndex], beta1, beta2, epsilon, adamT);
                UpdateWeightWithAdam(Wv[layerIndex], mWv[layerIndex], vWv[layerIndex], dWv[layerIndex], beta1, beta2, epsilon, adamT);
            });

            var frobeniusNorm = MatrixUtils.FrobeniusNorm(V);
            var learningRateReductionFactor = MatrixUtils.LearningRateReductionFactor(frobeniusNorm, 1.4d, 0.001d);
            Console.WriteLine($"{lstmName}: Frobenius norm: {frobeniusNorm}, learning rate reduction factor: {learningRateReductionFactor}");
            UpdateWeightWithAdam(V, mV, vV, dV, beta1, beta2, epsilon, adamT, learningRate * learningRateReductionFactor);
            UpdateWeightWithAdam(b, mb, vb, db, beta1, beta2, epsilon, adamT, learningRate * learningRateReductionFactor);

            UpdateWeightWithAdam(We, mWe, vWe, dWe, beta1, beta2, epsilon, adamT);
            UpdateWeightWithAdam(be, mbe, vbe, dbe, beta1, beta2, epsilon, adamT);
        }

        private void InitializeMoments()
        {
            mWi = MatrixUtils.InitializeZeroMatrix(numLayers, Wi[0].Length, Wi[0][0].Length);
            vWi = MatrixUtils.InitializeZeroMatrix(numLayers, Wi[0].Length, Wi[0][0].Length);
            mWf = MatrixUtils.InitializeZeroMatrix(numLayers, Wf[0].Length, Wf[0][0].Length);
            vWf = MatrixUtils.InitializeZeroMatrix(numLayers, Wf[0].Length, Wf[0][0].Length);
            mWc = MatrixUtils.InitializeZeroMatrix(numLayers, Wc[0].Length, Wc[0][0].Length);
            vWc = MatrixUtils.InitializeZeroMatrix(numLayers, Wc[0].Length, Wc[0][0].Length);
            mWo = MatrixUtils.InitializeZeroMatrix(numLayers, Wo[0].Length, Wo[0][0].Length);
            vWo = MatrixUtils.InitializeZeroMatrix(numLayers, Wo[0].Length, Wo[0][0].Length);

            mWq = MatrixUtils.InitializeZeroMatrix(numLayers, Wq[0].Length, Wq[0][0].Length);
            vWq = MatrixUtils.InitializeZeroMatrix(numLayers, Wq[0].Length, Wq[0][0].Length);
            mWk = MatrixUtils.InitializeZeroMatrix(numLayers, Wk[0].Length, Wk[0][0].Length);
            vWk = MatrixUtils.InitializeZeroMatrix(numLayers, Wk[0].Length, Wk[0][0].Length);
            mWv = MatrixUtils.InitializeZeroMatrix(numLayers, Wv[0].Length, Wv[0][0].Length);
            vWv = MatrixUtils.InitializeZeroMatrix(numLayers, Wv[0].Length, Wv[0][0].Length);

            mUi = MatrixUtils.InitializeZeroMatrix(numLayers, Ui[0].Length, Ui[0][0].Length);
            vUi = MatrixUtils.InitializeZeroMatrix(numLayers, Ui[0].Length, Ui[0][0].Length);
            mUf = MatrixUtils.InitializeZeroMatrix(numLayers, Uf[0].Length, Uf[0][0].Length);
            vUf = MatrixUtils.InitializeZeroMatrix(numLayers, Uf[0].Length, Uf[0][0].Length);
            mUc = MatrixUtils.InitializeZeroMatrix(numLayers, Uc[0].Length, Uc[0][0].Length);
            vUc = MatrixUtils.InitializeZeroMatrix(numLayers, Uc[0].Length, Uc[0][0].Length);
            mUo = MatrixUtils.InitializeZeroMatrix(numLayers, Uo[0].Length, Uo[0][0].Length);
            vUo = MatrixUtils.InitializeZeroMatrix(numLayers, Uo[0].Length, Uo[0][0].Length);

            mbi = MatrixUtils.InitializeZeroMatrix(numLayers, bi[0].Length, bi[0][0].Length);
            vbi = MatrixUtils.InitializeZeroMatrix(numLayers, bi[0].Length, bi[0][0].Length);
            mbf = MatrixUtils.InitializeZeroMatrix(numLayers, bf[0].Length, bf[0][0].Length);
            vbf = MatrixUtils.InitializeZeroMatrix(numLayers, bf[0].Length, bf[0][0].Length);
            mbc = MatrixUtils.InitializeZeroMatrix(numLayers, bc[0].Length, bc[0][0].Length);
            vbc = MatrixUtils.InitializeZeroMatrix(numLayers, bc[0].Length, bc[0][0].Length);
            mbo = MatrixUtils.InitializeZeroMatrix(numLayers, bo[0].Length, bo[0][0].Length);
            vbo = MatrixUtils.InitializeZeroMatrix(numLayers, bo[0].Length, bo[0][0].Length);
            mWe = MatrixUtils.InitializeZeroMatrix(We.Length, We[0].Length);
            vWe = MatrixUtils.InitializeZeroMatrix(We.Length, We[0].Length);
            mbe = MatrixUtils.InitializeZeroMatrix(be.Length, be[0].Length);
            vbe = MatrixUtils.InitializeZeroMatrix(be.Length, be[0].Length);
            mV = MatrixUtils.InitializeZeroMatrix(V.Length, V[0].Length);
            vV = MatrixUtils.InitializeZeroMatrix(V.Length, V[0].Length);
            mb = MatrixUtils.InitializeZeroMatrix(b.Length, b[0].Length);
            vb = MatrixUtils.InitializeZeroMatrix(b.Length, b[0].Length);
        }

        private void UpdateWeightWithAdam(double[][] W, double[][] mW, double[][] vW, double[][] gradient, double beta1, double beta2, double epsilon, int t, double? newLearningRate = null)
        {
            var lr = newLearningRate.HasValue ? newLearningRate.Value : learningRate;

            // Update biased first moment estimate
            mW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta1, mW), MatrixUtils.ScalarMultiply(1 - beta1, gradient));

            // Update biased second raw moment estimate
            vW = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta2, vW), MatrixUtils.ScalarMultiply(1 - beta2, MatrixUtils.HadamardProduct(gradient, gradient)));

            // Compute bias-corrected first moment estimate
            double[][] mW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, t)), mW);

            // Compute bias-corrected second raw moment estimate
            double[][] vW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, t)), vW);

            // Update weights
            for (int i = 0; i < W.Length; i++)
            {
                for (int j = 0; j < W[0].Length; j++)
                {
                    double weightReductionValue = lr * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);
                    W[i][j] -= weightReductionValue;
                }
            }
        }

        private IOperation ProcessOperation(OperationInfo operation, int timeStep = -1, int layerIndex = -1)
        {
            if (operation == null)
            {
                throw new ArgumentNullException(nameof(operation), $"The parameter {nameof(operation)} cannot be null.");
            }

            string id = operation.Id;
            string typeName = operation.Type;
            string[] inputs = operation.Inputs;

            System.Type operationType = System.Type.GetType($"{NAMESPACE}.{typeName}");
            if (operationType == null)
            {
                throw new InvalidOperationException($"Unsupported operation type: {typeName}");
            }

            IOperation op = (IOperation)operationType.GetMethod("Instantiate").Invoke(null, new object[] { (NeuralNetwork)this });

            op.OperationType = operationType;
            op.Inputs = inputs.ToList();
            string resultTo = operation.SetResultTo;
            if (resultTo != null)
            {
                op.ResultToName = resultTo;
            }

            string[] gradientResultTo = operation.GradientResultTo;
            if (gradientResultTo != null)
            {
                op.GradientDestinations = new object[gradientResultTo.Length];
                for (int i = 0; i < gradientResultTo.Length; ++i)
                {
                    if (gradientResultTo[i] != null)
                    {
                        op.GradientDestinations[i] = NameToValueFunc(gradientResultTo[i])(timeStep, layerIndex);
                    }
                }
            }

            if (priorOperation != null)
            {
                priorOperation.Next = op;
            }

            if (startOperation == null)
            {
                startOperation = op;
            }

            op.Id = id;

            priorOperation = op;

            return op;
        }

        private void ProcessAndAddOperation(OperationInfo operationInfo, int timeStepIndex, int layerIndex = -1)
        {
            IOperation op = ProcessOperation(operationInfo, timeStepIndex, layerIndex);
            op.TimeStepIndex = timeStepIndex;
            op.SpecificId = op.Id + "_" + timeStepIndex;

            if (layerIndex != -1)
            {
                op.LayerIndex = layerIndex;
                op.SpecificId += "_" + layerIndex;
            }

            operationsMap[op.SpecificId] = op;
        }

        private void CreateOperationsFromJson(string json)
        {
            operationsMap = new Dictionary<string, IOperation>();
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json);
            priorOperation = null;
            startOperation = null;

            for (int i = 0; i < numTimeSteps; ++i)
            {
                foreach (var timeStep in jsonArchitecture.TimeSteps)
                {

                    foreach (var start in timeStep.StartOperations)
                    {
                        ProcessAndAddOperation(start, i);
                    }

                    for (int j = 0; j < numLayers; ++j) {
                        foreach (var layer in timeStep.Layers)
                        {
                            foreach (var layerOp in layer.Operations)
                            {
                                ProcessAndAddOperation(layerOp, i, j);
                            }
                        }
                    }

                    foreach (var end in timeStep.EndOperations)
                    {
                        ProcessAndAddOperation(end, i);
                    }
                }
            }

            priorOperation = null;
        }

        public void SaveModel(string filePath)
        {
            MultiLayerLSTMParameters parameters = new MultiLayerLSTMParameters
            {
                Wi = Wi,
                Wf = Wf,
                Wc = Wc,
                Wo = Wo,
                Ui = Ui,
                Uf = Uf,
                Uc = Uc,
                Uo = Uo,
                bi = bi,
                bf = bf,
                bc = bc,
                bo = bo,
                be = be,
                We = We,
                V = V,
                b = b,
                Wq = Wq,
                Wk = Wk,
                Wv = Wv
            };

            var serialized = SerializeLSTMParameters(parameters);
            File.WriteAllText(filePath, serialized);
        }

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
            lstm.bi = parameters.bi;
            lstm.bf = parameters.bf;
            lstm.bc = parameters.bc;
            lstm.bo = parameters.bo;
            lstm.be = parameters.be;
            lstm.We = parameters.We;
            lstm.Wq = parameters.Wq;
            lstm.Wk = parameters.Wk;
            lstm.Wv = parameters.Wv;
            lstm.V = parameters.V;
            lstm.b = parameters.b;

            return lstm;
        }

        // Serialize LSTMParameters to a JSON string
        public static string SerializeLSTMParameters(MultiLayerLSTMParameters parameters)
        {
            string jsonString = JsonConvert.SerializeObject(parameters, Newtonsoft.Json.Formatting.Indented);
            return jsonString;
        }

        // Deserialize a JSON string back to LSTMParameters
        public static MultiLayerLSTMParameters DeserializeLSTMParameters(string jsonString)
        {
            MultiLayerLSTMParameters parameters = JsonConvert.DeserializeObject<MultiLayerLSTMParameters>(jsonString);
            return parameters;
        }

    }
}