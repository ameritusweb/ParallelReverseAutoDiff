// ------------------------------------------------------------------------------
// <copyright file="MazeNet.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ManagedCuda.BasicTypes;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD;

    /// <summary>
    /// A gated recurrent network.
    /// </summary>
    public class MazeNet : NeuralNetwork
    {
        private readonly int numTimeSteps;
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private StochasticAdamOptimizer adamOptimize;
        private GradientClipper clipper;

        private VGruNetwork.MazeNetwork gatedRecurrentNetwork;

        private List<IModelLayer> modelLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="MazeNet"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public MazeNet(int numTimeSteps, int numNodes, int numFeatures, int numLayers, double learningRate, double clipValue)
        {
            this.numTimeSteps = numTimeSteps;
            this.numFeatures = numFeatures;
            this.numNodes = numNodes;
            this.numLayers = numLayers;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
        }

        /// <summary>
        /// Gets a gated recurrent network.
        /// </summary>
        public VGruNetwork.MazeNetwork MazeNetwork => this.gatedRecurrentNetwork;

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            await this.gatedRecurrentNetwork.Initialize();
            this.gatedRecurrentNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
        }

        /// <summary>
        /// Gets the model layer.
        /// </summary>
        /// <returns>The model layer.</returns>
        public IModelLayer? GetModelLayer()
        {
            return this.gatedRecurrentNetwork.ModelLayers.FirstOrDefault();
        }

        /// <summary>
        /// Adjusts the learning rate.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public void AdjustLearningRate(double learningRate)
        {
            this.gatedRecurrentNetwork.Parameters.LearningRate = learningRate;
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            var initialAdamIteration = 1;
            var model = new VGruNetwork.MazeNetwork(this.numTimeSteps, this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
            model.Parameters.AdamIteration = initialAdamIteration;
            this.gatedRecurrentNetwork = model;
            this.adamOptimize = new StochasticAdamOptimizer(this.gatedRecurrentNetwork);
            this.clipper = this.gatedRecurrentNetwork.Utilities.GradientClipper;
            await this.gatedRecurrentNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.gatedRecurrentNetwork.ModelLayers).ToList();
        }

        /// <summary>
        /// Randomizes weights.
        /// </summary>
        public void RandomizeWeights()
        {
            foreach (var modelLayer in this.modelLayers)
            {
                modelLayer.RandomizeWeights();
            }
        }

        /// <summary>
        /// Save the weights to the save path.
        /// </summary>
        public void SaveWeights()
        {
            Guid guid = Guid.NewGuid();
            var dir = $"E:\\mazestore\\{guid}_{this.gatedRecurrentNetwork.Parameters.AdamIteration}";
            Directory.CreateDirectory(dir);
            int index = 0;
            foreach (var modelLayer in this.modelLayers)
            {
                modelLayer.SaveWeightsAndMomentsBinary(new FileInfo($"{dir}\\layer{index}"));
                index++;
            }
        }

        /// <summary>
        /// Apply the weights from the save path.
        /// </summary>
        public void ApplyWeights()
        {
            var guid = "650f3fb3-e67d-41bb-be2c-45ddd3ad4d58_3610";
            var dir = $"E:\\mazestore\\{guid}";
            for (int i = 0; i < this.modelLayers.Count; ++i)
            {
                var modelLayer = this.modelLayers[i];
                var file = new FileInfo($"{dir}\\layer{i}");
                modelLayer.LoadWeightsAndMomentsBinary(file);
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
            }
        }

        /// <summary>
        /// Apply the gradients to update the weights.
        /// </summary>
        public void ApplyGradients()
        {
            var clipper = this.clipper;
            clipper.Clip(this.modelLayers.ToArray());
            var adamOptimizer = this.adamOptimize;
            adamOptimizer.Optimize(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Train the network.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="targetAngles">The target angles.</param>
        /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
        public async Task Train(DeepMatrix input, double[] targetAngles)
        {
            int[,] structure = new int[7, 7];
            structure[3, 3] = 1;
            var depth = input.Depth;
            Matrix? previousInput = null;
            Matrix[] previousHiddenState = new Matrix[4];
            List<MazeComputationGraph> computationGraphs = new List<MazeComputationGraph>();
            for (int i = 0; i < depth; ++i)
            {
                var gruInput = input[i];
                var targetAngle = targetAngles[i];

                if (i == 0)
                {
                    var gruNet = this.gatedRecurrentNetwork;
                    gruNet.InitializeState();

                    gruNet.AutomaticForwardPropagate(gruInput, null);

                    computationGraphs.Add(gruNet.ComputationGraph);

                    var output = gruNet.Output;
                    previousHiddenState[0] = (Matrix)gruNet.HiddenState.ConcatItself(AppendDirection.VectorRight);
                    previousHiddenState[1] = (Matrix)gruNet.HiddenState.ConcatItself(AppendDirection.VectorLeft);
                    previousHiddenState[2] = (Matrix)gruNet.HiddenState.ConcatItself(AppendDirection.Up);
                    previousHiddenState[3] = (Matrix)gruNet.HiddenState.ConcatItself(AppendDirection.Down);

                    SquaredArclengthEuclideanLossOperation lossOp = SquaredArclengthEuclideanLossOperation.Instantiate(gruNet);
                    var loss = lossOp.Forward(output, targetAngle);
                    var gradient = lossOp.Backward();

                    await this.gatedRecurrentNetwork.AutomaticBackwardPropagate(gradient);

                    this.gatedRecurrentNetwork.UpdateModelLayers();

                    previousInput = gruInput;
                }
                else if (i == 1)
                {
                    var gruNet = this.gatedRecurrentNetwork;
                    structure[3, 4] = 1;

                    var appendedInput1 = gruInput.Append(previousInput!, AppendDirection.Left);

                    await gruNet.Reinitialize(structure);
                    gruNet.InitializeState();
                    gruNet.AutomaticForwardPropagate(appendedInput1, previousHiddenState[0]);
                    var output1 = gruNet.Output;
                    var hiddenState1 = gruNet.HiddenState;

                    SquaredArclengthEuclideanLossOperation lossOp1 = SquaredArclengthEuclideanLossOperation.Instantiate(gruNet);
                    var loss1 = lossOp1.Forward(output1, targetAngle);
                    var gradient1 = lossOp1.Backward();

                    await this.gatedRecurrentNetwork.AutomaticBackwardPropagate(gradient1);

                    this.gatedRecurrentNetwork.UpdateModelLayers();

                    structure[3, 4] = 0;
                    structure[3, 2] = 1;

                    var appendedInput2 = gruInput.Append(previousInput!, AppendDirection.Right);

                    await gruNet.Reinitialize(structure);
                    gruNet.InitializeState();
                    gruNet.AutomaticForwardPropagate(appendedInput2, previousHiddenState[1]);
                    var output2 = gruNet.Output;
                    var hiddenState2 = gruNet.HiddenState;

                    SquaredArclengthEuclideanLossOperation lossOp2 = SquaredArclengthEuclideanLossOperation.Instantiate(gruNet);
                    var loss2 = lossOp2.Forward(output2, targetAngle);
                    var gradient2 = lossOp2.Backward();

                    await this.gatedRecurrentNetwork.AutomaticBackwardPropagate(gradient2);

                    this.gatedRecurrentNetwork.UpdateModelLayers();
                }
            }
        }

        /*
        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="targetAngle">The target angle.</param>
        /// <param name="targetMagnitude">The target magnitude.</param>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public (Matrix Gradient, Matrix Output, Matrix Loss) Forward(DeepMatrix input, double targetAngle, double targetMagnitude)
        {
            var gruNet = this.gatedRecurrentNetwork;
            gruNet.InitializeState();
            gruNet.AutomaticForwardPropagate(input);
            var output = gruNet.Output;

            var numTimeSteps = gruNet.Parameters.NumTimeSteps;

            var result = output[numTimeSteps - 1];

            ContributionLossOperation contributionOp = ContributionLossOperation.Instantiate(gruNet);
            var loss = contributionOp.Forward(result, targetAngle);
            var gradient = contributionOp.Backward();

            // CascadingLossOperation cascadingLossOperation = CascadingLossOperation.Instantiate(gruNet);
            // var loss = cascadingLossOperation.Forward(result, targetAngle);
            // var gradient = cascadingLossOperation.Backward();

            // var matrix = VectorToMatrix.CreateLine(targetAngle, 11);

            // ElementwiseVectorContributionOperation contributionOp = ElementwiseVectorContributionOperation.Instantiate(gruNet);
            // var res = contributionOp.Forward(result, matrix);

            // SquaredArclengthEuclideanMagnitudeLossOperation4 lossOp = SquaredArclengthEuclideanMagnitudeLossOperation4.Instantiate(gruNet);
            // var loss = lossOp.Forward(res, targetAngle, targetMagnitude);
            // var gradientFinal = lossOp.Backward();

            // var gradientResult = contributionOp.Backward(gradientFinal);
            // var gradient = gradientResult.Item1 as Matrix;
            return (gradient ?? new Matrix(), result, loss);
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task<Matrix> Backward(Matrix gradientOfLossWrtOutput)
        {
            return await this.gatedRecurrentNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput);
        }
        */
    }
}
