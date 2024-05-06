// ------------------------------------------------------------------------------
// <copyright file="MusicNet.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD;

    /// <summary>
    /// A gated recurrent network.
    /// </summary>
    public class MusicNet : NeuralNetwork
    {
        private readonly int numTimeSteps;
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private StochasticAdamOptimizer adamOptimize;
        private GradientClipper clipper;

        private VGruNetwork.MusicNetwork gatedRecurrentNetwork;

        private List<IModelLayer> modelLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="MusicNet"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public MusicNet(int numTimeSteps, int numNodes, int numFeatures, int numLayers, double learningRate, double clipValue)
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
        public VGruNetwork.MusicNetwork MazeNetwork => this.gatedRecurrentNetwork;

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            this.gatedRecurrentNetwork.ResetMaze();
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
            var model = new VGruNetwork.MusicNetwork(this.numTimeSteps, this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
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
            var dir = $"E:\\musicstore\\{guid}_{this.gatedRecurrentNetwork.Parameters.AdamIteration}";
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
            var guid = "9ef3e964-308f-4d15-b93f-ebede401b996_463";
            var dir = $"E:\\musicstore\\{guid}";
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
            MusicGrid tileGrid = new MusicGrid(this.gatedRecurrentNetwork, input.ToArray().ToList(), targetAngles.ToList());
            await tileGrid.RunTimeSteps();
        }
    }
}
