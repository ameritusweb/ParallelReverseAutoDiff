using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class GlyphNet
    {
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private GlyphNetwork.GlyphNetwork GlyphNetwork;

        private List<IModelLayer> modelLayers;
        private readonly List<(string, string)> entities;
        private Matrix? prevOutputTwo;
        private readonly StochasticAdamOptimizer adamOptimize;

        /// <summary>
        /// Initializes a new instance of the <see cref="GlyphNetwork"/> class.
        /// </summary>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public GlyphNet(int numNodes, int numFeatures, int numLayers, double learningRate, double clipValue)
        {
            this.numFeatures = numFeatures;
            this.numNodes = numNodes;
            this.numLayers = numLayers;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.entities = new List<(string, string)>();
            this.GlyphNetwork = new GlyphNetwork.GlyphNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue, "glyphnet");
            this.adamOptimize = new StochasticAdamOptimizer(this.GlyphNetwork);
        }

        public GlyphNetwork.GlyphNetwork GraphAttentionNetwork => this.GlyphNetwork;

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            await this.GlyphNetwork.Initialize();
            this.GlyphNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
        }

        public IModelLayer? GetModelLayer()
        {
            return this.GlyphNetwork.ModelLayers.FirstOrDefault();
        }

        /// <summary>
        /// Adjusts the learning rate.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public void AdjustLearningRate(double learningRate)
        {
            this.GlyphNetwork.Parameters.LearningRate = learningRate;
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            var initialAdamIteration = 138;
            var model = new GlyphNetwork.GlyphNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue, "glyphnet");
            model.Parameters.AdamIteration = initialAdamIteration;
            this.GlyphNetwork = model;
            await this.GlyphNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.GlyphNetwork.ModelLayers).ToList();
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
            var dir = $"E:\\vnnstore\\glyph_{guid}_{this.GlyphNetwork.Parameters.AdamIteration}";
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
            var guid = "glyph_c83acea3-57af-4909-adab-0c75768549de_138";
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
            var clipper = this.GlyphNetwork.Utilities.GradientClipper;
            clipper.Clip(this.modelLayers.ToArray());
            var adamOptimizer = this.adamOptimize;
            adamOptimizer.Optimize(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public (Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix) Forward(Matrix input, Matrix rotationTargets, double targetAngle)
        {

            var gatNet = this.GlyphNetwork;
            gatNet.TargetAngle = targetAngle;
            gatNet.InitializeState();
            gatNet.RotationTargets.Replace(rotationTargets.ToArray());
            gatNet.AutomaticForwardPropagate(input);
            var output = gatNet.Output;
            var glyph = gatNet.Glyph;
            var targetedSum0 = gatNet.TargetedSum0;
            var targetedSum1 = gatNet.TargetedSum1;

            int maxMag0 = 0;
            int maxMag1 = 0;
            for (int i = 0; i < 15; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    if (rotationTargets[i, j] == 1)
                    {
                        maxMag1++;
                    }
                    else
                    {
                        maxMag0++;
                    }
                }
            }

            Console.WriteLine($"Max Mag 0: {maxMag0}, Max Mag 1: {maxMag1}");

            SquaredArclengthEuclideanLossOperation arclengthLoss0 = SquaredArclengthEuclideanLossOperation.Instantiate(gatNet);
            var loss0 = arclengthLoss0.Forward(targetedSum0, (3 * Math.PI) / 4d, 0);
            var gradient0 = arclengthLoss0.Backward();

            SquaredArclengthEuclideanLossOperation arclengthLoss1 = SquaredArclengthEuclideanLossOperation.Instantiate(gatNet);
            var loss1 = arclengthLoss1.Forward(targetedSum1, (1 * Math.PI) / 4d, 1);
            var gradient1 = arclengthLoss1.Backward();

            SquaredArclengthEuclideanMagnitudeLossOperation arclengthLoss = SquaredArclengthEuclideanMagnitudeLossOperation.Instantiate(gatNet);
            var loss = arclengthLoss.Forward(output, targetAngle, 225);
            var gradient = arclengthLoss.Backward();

            return (gradient, gradient0, gradient1, output, targetedSum0, targetedSum1, glyph, loss, loss0, loss1);
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task<Matrix> Backward(Matrix gradientOfLossWrtOutput, Matrix gradient0, Matrix gradient1)
        {
            return await this.GlyphNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput, gradient0, gradient1);
        }
    }
}
