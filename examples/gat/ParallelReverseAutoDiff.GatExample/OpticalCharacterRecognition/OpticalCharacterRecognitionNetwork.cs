//------------------------------------------------------------------------------
// <copyright file="OpticalCharacterRecognitionNetwork.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition
{
    using System;
    using System.IO;
    using ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition.RMAD;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Optical character recognition neural network.
    /// </summary>
    public class OpticalCharacterRecognitionNetwork
    {
        private readonly int numFeatures;
        private readonly int numNodes;
        private readonly int numLayers;
        private readonly double learningRate;
        private readonly double clipValue;

        private GraphAttentionNetwork.GraphAttentionNetwork graphAttentionNetwork;

        private List<IModelLayer> modelLayers;

        /// <summary>
        /// Initializes a new instance of the <see cref="OpticalCharacterRecognitionNetwork"/> class.
        /// </summary>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numFeatures">The number of features.</param>
        /// <param name="numLayers">The number of layers.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip Value.</param>
        public OpticalCharacterRecognitionNetwork(int numNodes, int numFeatures, int numLayers,double learningRate, double clipValue)
        {
            this.numFeatures = numFeatures;
            this.numNodes = numNodes;
            this.numLayers = numLayers;
            this.learningRate = learningRate;
            this.clipValue = clipValue;
            this.modelLayers = new List<IModelLayer>();
            this.graphAttentionNetwork = new GraphAttentionNetwork.GraphAttentionNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
        }

        public GraphAttentionNetwork.GraphAttentionNetwork GraphAttentionNetwork => this.graphAttentionNetwork;

        /// <summary>
        /// Reset the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Reset()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(this.modelLayers.ToArray());

            await this.graphAttentionNetwork.Initialize();
            this.graphAttentionNetwork.Parameters.AdamIteration++;

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true);
        }

        /// <summary>
        /// Adjusts the learning rate.
        /// </summary>
        /// <param name="learningRate">The learning rate.</param>
        public void AdjustLearningRate(double learningRate)
        {
            this.graphAttentionNetwork.Parameters.LearningRate = learningRate;
        }

        /// <summary>
        /// Initializes the model layers.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            var initialAdamIteration = 777;
            var model = new GraphAttentionNetwork.GraphAttentionNetwork(this.numLayers, this.numNodes, this.numFeatures, this.learningRate, this.clipValue);
            model.Parameters.AdamIteration = initialAdamIteration;
            this.graphAttentionNetwork = model;
            await this.graphAttentionNetwork.Initialize();
            this.modelLayers = this.modelLayers.Concat(this.graphAttentionNetwork.ModelLayers).ToList();
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
            Guid guid = Guid.NewGuid();
            var dir = $"E:\\gatstore\\{guid}_{this.graphAttentionNetwork.Parameters.AdamIteration}";
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
            var guid = "c791bf15-791d-45dc-aa77-fe5016707180_777";
            var dir = $"E:\\gatstore\\{guid}";
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
            var clipper = this.graphAttentionNetwork.Utilities.GradientClipper;
            clipper.Clip(this.modelLayers.ToArray());
            var adamOptimizer = new StochasticAdamOptimizer(this.graphAttentionNetwork);
            adamOptimizer.Optimize(this.modelLayers.ToArray());
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public (Matrix, Matrix, List<double>) Forward(Matrix input, double targetMax)
        {
            var gatNet = this.graphAttentionNetwork;
            gatNet.InitializeState();
            gatNet.AutomaticForwardPropagate(input);
            var output = gatNet.Output;
            var arrList = output[0].ToList();
            var rr = arrList.OrderByDescending(r => r).ToList();
            var scaled = ScaleValuesToMax(arrList, targetMax);
            var rrScaled = scaled.OrderByDescending(r => r).ToList();
            MeanSquaredErrorLossOperation lossOperation = MeanSquaredErrorLossOperation.Instantiate(this.graphAttentionNetwork);
            lossOperation.Forward(output, new Matrix(scaled.ToArray()));
            var gradient = lossOperation.Backward();
            return (gradient, output, rr);
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public double Forward2(Matrix input)
        {
            var gatNet = this.graphAttentionNetwork;
            gatNet.InitializeState();
            gatNet.AutomaticForwardPropagate(input);
            var output = gatNet.Output;

            var res = ComputeVariedSoftmax(output[0], 0.0002d);
            var rr = res.OrderByDescending(r => r).ToList();

            double max = Math.Round(rr.Max(), 3);
            var rrr = res.Where(x => Math.Round(x, 3) == max).ToList();

            return max;

            //List<int> indices = new List<int>();
            //for (int i = 0; i < 17; ++i)
            //{
            //    var ind = res.ToList().FindIndex(x => x == rr[i]);
            //    indices.Add(ind);
            //}
            //var ord = indices.OrderByDescending(x => x).ToList();

            //// Create a list to store the original numbers and their ranks
            //var numberRankPairs = new List<(int Number, int Rank)>();

            //// Iterate through the original list and find each number's rank
            //foreach (var number in indices)
            //{
            //    int rank = ord.IndexOf(number) + 1;
            //    numberRankPairs.Add((number, rank));
            //}

            //var rrrr = rr.Select(x => Math.Round(x, 3)).Distinct().ToList();
            //var scaled = ScaleValuesToMax(rrrr, 3d);


            //return numberRankPairs;
            //return rrr.ToArray();
            //var sum = rr.Sum();
            //var maxRounded = Math.Round(rr.Max(), 3);
            //var minRounded = Math.Round(rr.Min(), 3);
            //var indices = res.Select((r, i) => new { r, i }).Where(ri => Math.Round(ri.r, 3) == maxRounded).Select(ri => ri.i).ToList();
            //List<double> list = new List<double>();
            //foreach (var index in indices)
            //{
            //    list.Add(res[index]);
            //}

            //return output;
            //CategoricalCrossEntropyLossOperation lossOperation = new CategoricalCrossEntropyLossOperation();
            //lossOperation.Forward(output, this.maze.ToTrueLabel(output.Cols));
            //var gradientOfLoss = lossOperation.Backward();

            //return gradientOfLoss;
        }

        public List<double> ScaleValuesToMax(List<double> values, double newMax)
        {
            if (values == null || !values.Any())
                throw new ArgumentException("Values list cannot be null or empty.");

            double currentMax = values.Max();
            if (currentMax == 0)
                return values; // Avoid division by zero.

            return values.Select(value => value / currentMax * newMax).ToList();
        }

        public double[] ComputeVariedSoftmax(double[] x, double temperature)
        {
            double[] softmax = new double[x.Length];
            double sumExp = x.Sum(xi => Math.Exp(xi / temperature));

            for (int i = 0; i < x.Length; i++)
            {
                softmax[i] = Math.Exp(x[i] / temperature) / sumExp;
            }

            double scaleFactor = Math.Sqrt(x.Length) / softmax.Sum();
            return softmax.Select(s => s * scaleFactor).ToArray();
        }

        /// <summary>
        /// The backward pass through the computation graph.
        /// </summary>
        /// <param name="gradientOfLossWrtOutput">The gradient of the loss wrt the output.</param>
        /// <returns>A task.</returns>
        public async Task<Matrix> Backward(Matrix gradientOfLossWrtOutput)
        {
            return await this.graphAttentionNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput);
        }
    }
}
