using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class VectorFieldNet
    {
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private VectorFieldNetwork.VectorFieldNetwork vectorFieldNetwork;

        private List<IModelLayer> modelLayers;
        private List<(string, string)> entities;
        private Matrix? prevOutputTwo;
        private StochasticAdamOptimizer adamOptimize;

        /// <summary>
        /// Initializes a new instance of the <see cref="VectorFieldNetwork"/> class.
        /// </summary>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public VectorFieldNet(int numNodes, int numFeatures, int numLayers, double learningRate, double clipValue)
        {
            this.numFeatures = numFeatures;
            this.numNodes = numNodes;
            this.numLayers = numLayers;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.entities = new List<(string, string)>();
            this.vectorFieldNetwork = new VectorFieldNetwork.VectorFieldNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
            this.adamOptimize = new StochasticAdamOptimizer(this.vectorFieldNetwork);
        }

        public VectorFieldNetwork.VectorFieldNetwork GraphAttentionNetwork => this.vectorFieldNetwork;

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            await this.vectorFieldNetwork.Initialize();
            this.vectorFieldNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
        }

        public IModelLayer? GetModelLayer()
        {
            return this.vectorFieldNetwork.ModelLayers.FirstOrDefault();
        }

        /// <summary>
        /// Adjusts the learning rate.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public void AdjustLearningRate(double learningRate)
        {
            this.vectorFieldNetwork.Parameters.LearningRate = learningRate;
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            var initialAdamIteration = 1;
            var model = new VectorFieldNetwork.VectorFieldNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
            model.Parameters.AdamIteration = initialAdamIteration;
            this.vectorFieldNetwork = model;
            await this.vectorFieldNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.vectorFieldNetwork.ModelLayers).ToList();
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
            var dir = $"E:\\vnnstore\\field_{guid}_{this.vectorFieldNetwork.Parameters.AdamIteration}";
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
            var guid = "7be999f9-b2ef-4b42-88bd-575162097b23_8714";
            var dir = $"E:\\vnnstore\\field_{guid}";
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
            var clipper = this.vectorFieldNetwork.Utilities.GradientClipper;
            clipper.Clip(this.modelLayers.ToArray());
            var adamOptimizer = this.adamOptimize;
            adamOptimizer.Optimize(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public (Matrix, Matrix, Matrix) Forward(Matrix input, double targetAngle, double oppositeAngle)
        {

            var gatNet = this.vectorFieldNetwork;
            gatNet.TargetAngle = targetAngle;
            gatNet.OppositeAngle = oppositeAngle;
            gatNet.InitializeState();
            gatNet.AutomaticForwardPropagate(input);
            var output = gatNet.Output;

            SquaredArclengthLossOperation arclengthLoss = SquaredArclengthLossOperation.Instantiate(gatNet);
            var loss = arclengthLoss.Forward(output, targetAngle);
            var gradient = arclengthLoss.Backward();

            return (gradient, output, loss);
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task<Matrix> Backward(Matrix gradientOfLossWrtOutput)
        {
            return await this.vectorFieldNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput);
        }
    }
}
