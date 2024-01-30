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
        private List<(string, string)> entities;
        private Matrix? prevOutputTwo;
        private List<string> wrongs;
        private List<string> rights;
        private List<double> diffStore;
        private List<double> sameStore;
        private double totalDiff = 0;
        private double numberOfDiffs = 0;

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
            this.entities = new List<(string, string)>();
            this.wrongs = new List<string>();
            this.rights = new List<string>();
            this.diffStore = new List<double>();
            this.sameStore = new List<double>();
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
            var initialAdamIteration = 6064;
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
            var guid = "3a98ff3c-69af-4043-bc02-412456497116_6064";
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
            adamOptimizer.Optimize(new IModelLayer[] { this.modelLayers.LastOrDefault() });
        }

        /// <summary>
        /// Make a forward pass through the computation graph.
        /// </summary>
        /// <returns>The gradient of the loss wrt the output.</returns>
        public (Matrix, Matrix, List<double>) Forward(Matrix input, double targetMax, string f1, string f2)
        {
            string char1 = f1.Substring(0, 1);
            string char2 = f2.Substring(0, 1);

            var gatNet = this.graphAttentionNetwork;
            gatNet.InitializeState();
            gatNet.AutomaticForwardPropagate(input);
            var output = gatNet.Output;
            var outputTwo = gatNet.OutputTwo;
            var outputLeft = gatNet.OutputLeft;
            var outputRight = gatNet.OutputRight;
            var preOutputAdd = gatNet.PreOutputAdd;

            var maxDiff = Math.Abs(outputLeft[0].Max() - outputRight[0].Max());

            if (maxDiff < 0.001d && preOutputAdd.Sum() > 30d)
            {
                if (char1 != char2)
                {
                    if (sameStore.Any() && maxDiff < sameStore.LastOrDefault())
                    {
                        wrongs.Add($"1: {f1} {f2}");
                    }
                    
                    diffStore.Add(maxDiff);
                } else
                {
                    if (!diffStore.Any() || (diffStore.Any() && maxDiff < diffStore.LastOrDefault()))
                    {
                        rights.Add($"2: {f1} {f2}");
                        Console.WriteLine("Score: " + rights.Count + " " + wrongs.Count);
                    }

                    sameStore.Add(maxDiff);
                }
                if (rights.Count > 30 || wrongs.Count > 30)
                {
                    rights.Clear();
                    wrongs.Clear();
                    diffStore.Clear();
                    sameStore.Clear();
                }
            } else
            {
                if (char1 != char2)
                {
                    diffStore.Add(maxDiff);
                }
                else
                {
                    sameStore.Add(maxDiff);
                }
            }

            if (char1 == char2 && outputTwo[0][0] > outputTwo[0][1])
            {
                var avg = Math.Abs(outputTwo[0][0] - outputTwo[0][1]);
                totalDiff += avg;
                numberOfDiffs += 1d;
                var aa = totalDiff / numberOfDiffs;
                var aaa = aa * Math.Exp(aa);
                MeanSquaredErrorLossOperation lossOperation2 = MeanSquaredErrorLossOperation.Instantiate(this.graphAttentionNetwork);
                lossOperation2.Forward(outputTwo, new Matrix(new double[][] { new double[] { outputTwo[0][0] - (aaa * 100d), outputTwo[0][1] + (aaa * 100d) } }));
                var gradTwo = lossOperation2.Backward();
                return (gradTwo, outputTwo, new List<double>());
            } else if (char1 != char2 && outputTwo[0][0] < outputTwo[0][1])
            {
                var avg = Math.Abs(outputTwo[0][0] - outputTwo[0][1]);
                totalDiff += avg;
                numberOfDiffs += 1d;
                var aa = totalDiff / numberOfDiffs;
                var aaa = aa * Math.Exp(aa);
                MeanSquaredErrorLossOperation lossOperation2 = MeanSquaredErrorLossOperation.Instantiate(this.graphAttentionNetwork);
                lossOperation2.Forward(outputTwo, new Matrix(new double[][] { new double[] { outputTwo[0][0] + (aaa * 100d), outputTwo[0][1] - (aaa * 100d) } }));
                var gradTwo = lossOperation2.Backward();
                return (gradTwo, outputTwo, new List<double>());
            } else
            {
                if (char1 == char2 && (outputTwo[0][1] / outputTwo[0][0]) < 2d)
                {
                    var avg = Math.Abs(outputTwo[0][0] - outputTwo[0][1]);
                    totalDiff += avg;
                    numberOfDiffs += 1d;
                    var aaa = totalDiff / numberOfDiffs;
                    var ccc = (Math.Exp(aaa) - 1);
                    var aa = Math.Max(0d, aaa * (1 - ccc));
                    MeanSquaredErrorLossOperation lossOperation2 = MeanSquaredErrorLossOperation.Instantiate(this.graphAttentionNetwork);
                    lossOperation2.Forward(outputTwo, new Matrix(new double[][] { new double[] { outputTwo[0][0] - aa, outputTwo[0][1] + aa } }));
                    var gradTwo = lossOperation2.Backward();
                    return (gradTwo, outputTwo, new List<double>());
                } else if (char1 != char2 && (outputTwo[0][0] / outputTwo[0][1]) < 2d)
                {
                    var avg = Math.Abs(outputTwo[0][0] - outputTwo[0][1]);
                    totalDiff += avg;
                    numberOfDiffs += 1d;
                    var aaa = totalDiff / numberOfDiffs;
                    var ccc = (Math.Exp(aaa) - 1);
                    var aa = Math.Max(0d, aaa * (1 - ccc));
                    MeanSquaredErrorLossOperation lossOperation2 = MeanSquaredErrorLossOperation.Instantiate(this.graphAttentionNetwork);
                    lossOperation2.Forward(outputTwo, new Matrix(new double[][] { new double[] { outputTwo[0][0] + aa, outputTwo[0][1] - aa } }));
                    var gradTwo = lossOperation2.Backward();
                    return (gradTwo, outputTwo, new List<double>());
                }
                return (new Matrix(new double[][] { new double[] { 0.0d, 0.0d } }), outputTwo, new List<double>());
            }
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
        public async Task<Matrix> Backward(Matrix gradientOfLossWrtOutput, bool outputTwo)
        {
            return await this.graphAttentionNetwork.AutomaticBackwardPropagate(gradientOfLossWrtOutput, outputTwo);
        }
    }
}
