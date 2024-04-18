// ------------------------------------------------------------------------------
// <copyright file="SpatialNet.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD;

    /// <summary>
    /// A gated recurrent network.
    /// </summary>
    public class SpatialNet : NeuralNetwork
    {
        private readonly int numTimeSteps;
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private readonly StochasticAdamOptimizer adamOptimize;
        private readonly DynamicGradientClipper clipper;

        private VGruNetwork.SpatialNetwork gatedRecurrentNetwork;

        private List<IModelLayer> modelLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="SpatialNet"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public SpatialNet(int numTimeSteps, int numNodes, int numFeatures, int numLayers, double learningRate, double clipValue)
        {
            this.numTimeSteps = numTimeSteps;
            this.numFeatures = numFeatures;
            this.numNodes = numNodes;
            this.numLayers = numLayers;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.gatedRecurrentNetwork = new VGruNetwork.SpatialNetwork(this.numTimeSteps, this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
            this.adamOptimize = new StochasticAdamOptimizer(this.gatedRecurrentNetwork);
            this.clipper = new DynamicGradientClipper(this.gatedRecurrentNetwork);
        }

        /// <summary>
        /// Gets a gated recurrent network.
        /// </summary>
        public VGruNetwork.SpatialNetwork SpatialNetwork => this.gatedRecurrentNetwork;

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
            var model = new VGruNetwork.SpatialNetwork(this.numTimeSteps, this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
            model.Parameters.AdamIteration = initialAdamIteration;
            this.gatedRecurrentNetwork = model;
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
            var dir = $"E:\\vgrustore\\spatial_{guid}_{this.gatedRecurrentNetwork.Parameters.AdamIteration}";
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
            var guid = "spatial_650f3fb3-e67d-41bb-be2c-45ddd3ad4d58_3610";
            var dir = $"E:\\vgrustore\\{guid}";
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
        /// <param name="cGradient">The cascading gradient.</param>
        /// <returns>A task.</returns>
        public async Task<Matrix> Backward(Matrix gradientOfLossWrtOutput, Matrix? cGradient)
        {
            if (cGradient != null)
            {
                for (int i = 0; i < 11; ++i)
                {
                    for (int j = 0; j < 11; ++j)
                    {
                        gradientOfLossWrtOutput[i, j] = cGradient[i, j];
                    }
                }
            }

            return await this.gatedRecurrentNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput);
        }
    }
}
