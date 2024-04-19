// ------------------------------------------------------------------------------
// <copyright file="FusionNet.cs" author="ameritusweb" date="12/18/2023">
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
    public class FusionNet : NeuralNetwork
    {
        private readonly int numTimeSteps;
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private readonly StochasticAdamOptimizer adamOptimize;

        private VGruNetwork.SpatialNetwork gatedRecurrentNetwork;
        private VGruNetwork.VectorFieldNetwork vectorFieldNetwork;

        private List<IModelLayer> gruLayers;
        private List<IModelLayer> vnnLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="FusionNet"/> class.
        /// </summary>
        /// <param name="numTimeSteps">The number of time steps.</param>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public FusionNet(int numTimeSteps, int numNodes, int numFeatures, int numLayers, double learningRate, double clipValue)
        {
            this.numTimeSteps = numTimeSteps;
            this.numFeatures = numFeatures;
            this.numNodes = numNodes;
            this.numLayers = numLayers;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.gruLayers = new List<IModelLayer>();
            this.vnnLayers = new List<IModelLayer>();
            this.gatedRecurrentNetwork = new VGruNetwork.SpatialNetwork(this.numTimeSteps, this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
            this.vectorFieldNetwork = new VGruNetwork.VectorFieldNetwork(2, 33, 440, this.learningRate, this.clipValue);
            this.adamOptimize = new StochasticAdamOptimizer(this.gatedRecurrentNetwork);
        }

        /// <summary>
        /// Gets a gated recurrent network.
        /// </summary>
        public VGruNetwork.SpatialNetwork SpatialNetwork => this.gatedRecurrentNetwork;

        /// <summary>
        /// Gets a vector field network.
        /// </summary>
        public VGruNetwork.VectorFieldNetwork VectorFieldNetwork => this.vectorFieldNetwork;

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.gruLayers.ToArray());
            clearer.Clear(this.vnnLayers.ToArray());

            await this.gatedRecurrentNetwork.Initialize();
            await this.vectorFieldNetwork.Initialize();
            this.gatedRecurrentNetwork.Parameters.AdamIteration++;
            this.vectorFieldNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
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
            this.gruLayers = this.gruLayers.Concat(this.gatedRecurrentNetwork.ModelLayers).ToList();

            var model2 = new VGruNetwork.VectorFieldNetwork(2, 33, 440, this.learningRate, this.clipValue);
            model2.Parameters.AdamIteration = initialAdamIteration;
            this.vectorFieldNetwork = model2;
            await this.vectorFieldNetwork.Initialize();
            this.vnnLayers = this.vnnLayers.Concat(this.vectorFieldNetwork.ModelLayers).ToList();
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
            foreach (var modelLayer in this.gruLayers)
            {
                modelLayer.SaveWeightsAndMomentsBinary(new FileInfo($"{dir}\\layer{index}"));
                index++;
            }

            var dir2 = $"E:\\vgrustore\\vectorfield_{guid}_{this.gatedRecurrentNetwork.Parameters.AdamIteration}";
            Directory.CreateDirectory(dir2);
            int index2 = 0;
            foreach (var modelLayer in this.vnnLayers)
            {
                modelLayer.SaveWeightsAndMomentsBinary(new FileInfo($"{dir2}\\layer{index2}"));
                index2++;
            }
        }

        /// <summary>
        /// Apply the weights from the save path.
        /// </summary>
        public void ApplyWeights()
        {
            var gg = "650f3fb3-e67d-41bb-be2c-45ddd3ad4d58";
            var ii = 1;
            var guid = $"spatial_{gg}_{ii}";
            var dir = $"E:\\vgrustore\\{guid}";
            for (int i = 0; i < this.gruLayers.Count; ++i)
            {
                var modelLayer = this.gruLayers[i];
                var file = new FileInfo($"{dir}\\layer{i}");
                modelLayer.LoadWeightsAndMomentsBinary(file);
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
            }

            var guid2 = $"vectorfield_{gg}_{ii}";
            var dir2 = $"E:\\vgrustore\\{guid2}";
            for (int i = 0; i < this.vnnLayers.Count; ++i)
            {
                var modelLayer = this.vnnLayers[i];
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
            clipper.Clip(this.gruLayers.ToArray());
            clipper.Clip(this.vnnLayers.ToArray());
            var adamOptimizer = this.adamOptimize;
            adamOptimizer.Optimize(this.gruLayers.ToArray());
            adamOptimizer.Optimize(this.vnnLayers.ToArray());
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

            Matrix largeMatrix = new Matrix(33, 44);

            // Fill the large matrix with small matrices
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 4; col++)
                {
                    int index = (row * 4) + col;
                    Matrix small = output[index];
                    for (int i = 0; i < small.Rows; i++)
                    {
                        for (int j = 0; j < small.Cols; j++)
                        {
                            largeMatrix[(row * 11) + i, (col * 11) + j] = small[i, j];
                        }
                    }
                }
            }

            var vnnNet = this.vectorFieldNetwork;
            vnnNet.InitializeState();
            vnnNet.AutomaticForwardPropagate(largeMatrix);
            var result = vnnNet.Output;

            SquaredArclengthEuclideanLossOperation lossOp = SquaredArclengthEuclideanLossOperation.Instantiate(vnnNet);
            var loss = lossOp.Forward(result, targetAngle);
            var gradient = lossOp.Backward();

            return (gradient ?? new Matrix(), result, loss);
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task<Matrix> Backward(Matrix gradientOfLossWrtOutput)
        {
            var gradient = await this.vectorFieldNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput);
            var gruGradient = await this.gatedRecurrentNetwork.AutomaticBackwardPropagate(gradient);
            return gruGradient;
        }
    }
}
