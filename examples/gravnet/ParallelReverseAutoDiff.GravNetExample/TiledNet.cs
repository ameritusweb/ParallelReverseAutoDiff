using ParallelReverseAutoDiff.GravNetExample.Common;
using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class TiledNet
    {
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private TiledNetwork.TiledNetwork TiledNetwork;

        private List<IModelLayer> modelLayers;
        private readonly List<(string, string)> entities;
        private Matrix? prevOutputTwo;
        private readonly StochasticAdamOptimizer adamOptimize;

        /// <summary>
        /// Initializes a new instance of the <see cref="TiledNetwork"/> class.
        /// </summary>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public TiledNet(int numNodes, int numFeatures, int numLayers, double learningRate, double clipValue)
        {
            this.numFeatures = numFeatures;
            this.numNodes = numNodes;
            this.numLayers = numLayers;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.entities = new List<(string, string)>();
            this.TiledNetwork = new TiledNetwork.TiledNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue, "tilednet");
            this.adamOptimize = new StochasticAdamOptimizer(this.TiledNetwork);
        }

        public TiledNetwork.TiledNetwork GraphAttentionNetwork => this.TiledNetwork;

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            await this.TiledNetwork.Initialize();
            this.TiledNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
        }

        public IModelLayer? GetModelLayer()
        {
            return this.TiledNetwork.ModelLayers.FirstOrDefault();
        }

        /// <summary>
        /// Adjusts the learning rate.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public void AdjustLearningRate(double learningRate)
        {
            this.TiledNetwork.Parameters.LearningRate = learningRate;
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            var initialAdamIteration = 33;
            var model = new TiledNetwork.TiledNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue, "tilednet");
            model.Parameters.AdamIteration = initialAdamIteration;
            this.TiledNetwork = model;
            await this.TiledNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.TiledNetwork.ModelLayers).ToList();
        }

        /// <summary>
        /// Randomizes weights
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
            this.adamOptimize.Reset();
            Guid guid = Guid.NewGuid();
            var dir = $"E:\\vnnstore\\tiled_{guid}_{this.TiledNetwork.Parameters.AdamIteration}";
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
            var guid = "tiled_adf4a823-dbde-49c6-9505-c751f8097ffc_33";
            var dir = $"E:\\vnnstore\\{guid}";
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
            var clipper = this.TiledNetwork.Utilities.GradientClipper;
            clipper.Clip(this.modelLayers.ToArray());
            var adamOptimizer = this.adamOptimize;
            adamOptimizer.Optimize(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public (Matrix, (Matrix, Matrix)[,]) Forward(Matrix input, double[,] percentages)
        {

            var gatNet = this.TiledNetwork;
            //gatNet.TargetAngle = targetAngle;
            gatNet.InitializeState();
            //gatNet.RotationTargets.Replace(rotationTargets.ToArray());
            gatNet.AutomaticForwardPropagate(input);
            var output = gatNet.Output;

            var sections = CommonMatrixUtils.BreakIntoSectionsExactly(output, 8);
            (Matrix, Matrix)[,] lossAndGradient = new (Matrix, Matrix)[8, 8];
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 8; ++j)
                {
                    Matrix oo = sections[i, j];
                    var perc = percentages[i, j];
                    if (perc > 0.4d)
                    {
                        SquaredArclengthEuclideanLossOperation arclengthLoss = SquaredArclengthEuclideanLossOperation.Instantiate(gatNet);
                        var loss = arclengthLoss.Forward(oo, 3 * Math.PI / 4);
                        var gradient = arclengthLoss.Backward();
                        lossAndGradient[i, j] = (loss, gradient);
                    } else
                    {
                        SquaredArclengthEuclideanLossOperation arclengthLoss = SquaredArclengthEuclideanLossOperation.Instantiate(gatNet);
                        var loss = arclengthLoss.Forward(oo, Math.PI / 4);
                        var gradient = arclengthLoss.Backward();
                        lossAndGradient[i, j] = (loss, gradient);
                    }
                }
            }

            return (output, lossAndGradient);
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task<Matrix> Backward((Matrix, Matrix)[,] lossAndGradient)
        {
            return await this.TiledNetwork.AutomaticBackwardPropagate(lossAndGradient);
        }
    }
}
