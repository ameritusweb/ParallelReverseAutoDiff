//------------------------------------------------------------------------------
// <copyright file="DirectedAdamOptimizer.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Collections.Concurrent;
    using System.Threading.Tasks;

    /// <summary>
    /// An Adam optimizer.
    /// </summary>
    public class DirectedAdamOptimizer
    {
        private readonly NeuralNetwork network;
        private readonly ConcurrentDictionary<Matrix, HashSet<string>> matrixMap;
        private readonly ConcurrentDictionary<Matrix, double> scalingMap;
        private readonly ConcurrentDictionary<Matrix, HashSet<string>> e1Map;
        private readonly ConcurrentDictionary<Matrix, HashSet<string>> e2Map;
        private readonly ConcurrentDictionary<Matrix, HashSet<string>> intersectMap;

        private bool switchGradients;

        /// <summary>
        /// Initializes a new instance of the <see cref="DirectedAdamOptimizer"/> class.
        /// </summary>
        /// <param name="network">The neural network.</param>
        /// <param name="switchGradients">Switch the gradients.</param>
        public DirectedAdamOptimizer(NeuralNetwork network, bool switchGradients)
        {
            this.network = network;
            this.switchGradients = switchGradients;
            this.matrixMap = new ConcurrentDictionary<Matrix, HashSet<string>>();
            this.scalingMap = new ConcurrentDictionary<Matrix, double>();
            this.e1Map = new ConcurrentDictionary<Matrix, HashSet<string>>();
            this.e2Map = new ConcurrentDictionary<Matrix, HashSet<string>>();
            this.intersectMap = new ConcurrentDictionary<Matrix, HashSet<string>>();
        }

        /// <summary>
        /// Gets or sets a value indicating whether to switch the gradients.
        /// </summary>
        public bool SwitchGradients
        {
            get
            {
                return this.switchGradients;
            }

            set
            {
                this.switchGradients = value;
            }
        }

        /// <summary>
        /// Optimize the layers.
        /// </summary>
        /// <param name="layers">The layers to optimize.</param>
        public void Optimize(IModelLayer[] layers)
        {
            this.matrixMap.Clear();
            this.scalingMap.Clear();
            Parallel.For(0, layers.Length, i =>
            {
                var layer = layers[i];
                var identifiers = layer.Identifiers;
                for (int j = 0; j < identifiers.Count; ++j)
                {
                    var identifier = identifiers[j];
                    var weight = layer[identifier, ModelElementType.Weight];
                    var firstMoment = layer[identifier, ModelElementType.FirstMoment];
                    var secondMoment = layer[identifier, ModelElementType.SecondMoment];
                    var gradient = layer[identifier, ModelElementType.Gradient];
                    var dimensions = layer.Dimensions(identifier) ?? throw new InvalidOperationException("Dimensions cannot be null.");
                    switch (dimensions.Length)
                    {
                        case 2:
                            {
                                var weightMatrix = weight as Matrix ?? throw new InvalidOperationException("Weight cannot be null.");
                                var firstMomentMatrix = firstMoment as Matrix ?? throw new InvalidOperationException("First moment cannot be null.");
                                var secondMomentMatrix = secondMoment as Matrix ?? throw new InvalidOperationException("Second moment cannot be null.");
                                var gradientMatrix = gradient as Matrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                this.UpdateWeightWithAdam(identifier, weightMatrix, firstMomentMatrix, secondMomentMatrix, gradientMatrix, this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon, out HashSet<string> critical);
                                break;
                            }

                        case 3:
                            {
                                var weightMatrix = weight as DeepMatrix ?? throw new InvalidOperationException("Weight cannot be null.");
                                var firstMomentMatrix = firstMoment as DeepMatrix ?? throw new InvalidOperationException("First moment cannot be null.");
                                var secondMomentMatrix = secondMoment as DeepMatrix ?? throw new InvalidOperationException("Second moment cannot be null.");
                                var gradientMatrix = gradient as DeepMatrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                for (int d = 0; d < dimensions[0]; ++d)
                                {
                                    this.UpdateWeightWithAdam(identifier, weightMatrix[d], firstMomentMatrix[d], secondMomentMatrix[d], gradientMatrix[d], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon, out HashSet<string> critical);
                                }

                                break;
                            }

                        case 4:
                            {
                                var weightMatrix = weight as DeepMatrix[] ?? throw new InvalidOperationException("Weight cannot be null.");
                                var firstMomentMatrix = firstMoment as DeepMatrix[] ?? throw new InvalidOperationException("First moment cannot be null.");
                                var secondMomentMatrix = secondMoment as DeepMatrix[] ?? throw new InvalidOperationException("Second moment cannot be null.");
                                var gradientMatrix = gradient as DeepMatrix[] ?? throw new InvalidOperationException("Gradient cannot be null.");
                                for (int d1 = 0; d1 < dimensions[0]; ++d1)
                                {
                                    for (int d2 = 0; d2 < dimensions[1]; ++d2)
                                    {
                                        this.UpdateWeightWithAdam(identifier, weightMatrix[d1][d2], firstMomentMatrix[d1][d2], secondMomentMatrix[d1][d2], gradientMatrix[d1][d2], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon, out HashSet<string> critical);
                                    }
                                }

                                break;
                            }
                    }
                }
            });
        }

        /// <summary>
        /// Revert the last update. Be sure to decrement the AdamIteration before calling this function.
        /// </summary>
        /// <param name="layers">The model layers.</param>
        public void Revert(IModelLayer[] layers)
        {
            Parallel.For(0, layers.Length, i =>
            {
                var layer = layers[i];
                var identifiers = layer.Identifiers;
                for (int j = 0; j < identifiers.Count; ++j)
                {
                    var identifier = identifiers[j];
                    var weight = layer[identifier, ModelElementType.Weight];
                    var firstMoment = layer[identifier, ModelElementType.FirstMoment];
                    var secondMoment = layer[identifier, ModelElementType.SecondMoment];
                    var gradient = layer[identifier, ModelElementType.Gradient];
                    var dimensions = layer.Dimensions(identifier) ?? throw new InvalidOperationException("Dimensions cannot be null.");
                    switch (dimensions.Length)
                    {
                        case 2:
                            {
                                var weightMatrix = weight as Matrix ?? throw new InvalidOperationException("Weight cannot be null.");
                                var firstMomentMatrix = firstMoment as Matrix ?? throw new InvalidOperationException("First moment cannot be null.");
                                var secondMomentMatrix = secondMoment as Matrix ?? throw new InvalidOperationException("Second moment cannot be null.");
                                var gradientMatrix = gradient as Matrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                this.ReverseAdamUpdate(weightMatrix, firstMomentMatrix, secondMomentMatrix, gradientMatrix, this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon);
                                break;
                            }

                        case 3:
                            {
                                var weightMatrix = weight as DeepMatrix ?? throw new InvalidOperationException("Weight cannot be null.");
                                var firstMomentMatrix = firstMoment as DeepMatrix ?? throw new InvalidOperationException("First moment cannot be null.");
                                var secondMomentMatrix = secondMoment as DeepMatrix ?? throw new InvalidOperationException("Second moment cannot be null.");
                                var gradientMatrix = gradient as DeepMatrix ?? throw new InvalidOperationException("Gradient cannot be null.");
                                for (int d = 0; d < dimensions[0]; ++d)
                                {
                                    this.ReverseAdamUpdate(weightMatrix[d], firstMomentMatrix[d], secondMomentMatrix[d], gradientMatrix[d], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon);
                                }

                                break;
                            }

                        case 4:
                            {
                                var weightMatrix = weight as DeepMatrix[] ?? throw new InvalidOperationException("Weight cannot be null.");
                                var firstMomentMatrix = firstMoment as DeepMatrix[] ?? throw new InvalidOperationException("First moment cannot be null.");
                                var secondMomentMatrix = secondMoment as DeepMatrix[] ?? throw new InvalidOperationException("Second moment cannot be null.");
                                var gradientMatrix = gradient as DeepMatrix[] ?? throw new InvalidOperationException("Gradient cannot be null.");
                                for (int d1 = 0; d1 < dimensions[0]; ++d1)
                                {
                                    for (int d2 = 0; d2 < dimensions[1]; ++d2)
                                    {
                                        this.ReverseAdamUpdate(weightMatrix[d1][d2], firstMomentMatrix[d1][d2], secondMomentMatrix[d1][d2], gradientMatrix[d1][d2], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon);
                                    }
                                }

                                break;
                            }
                    }
                }
            });
        }

        private void ReverseAdamUpdate(Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon)
        {
            var critical = this.matrixMap[w];
            var scalingFactor = this.scalingMap[w];

            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    if (critical.Contains($"{i} {j}"))
                    {
                        gradient[i][j] = 0d; // *= scalingFactor;
                    }
                    else
                    {
                        gradient[i][j] *= scalingFactor;
                    }
                }
            }

            // Compute bias-corrected first moment estimate
            Matrix mW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, this.network.Parameters.AdamIteration)), mW);

            // Compute bias-corrected second raw moment estimate
            Matrix vW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, this.network.Parameters.AdamIteration)), vW);

            // Revert weights
            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    double weightReductionValue = this.network.Parameters.LearningRate * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);

                    var grad = gradient[i][j];

                    w[i][j] += weightReductionValue;  // adding here to reverse the update
                }
            }

            // Revert first moment
            for (int i = 0; i < mW.Length; i++)
            {
                for (int j = 0; j < mW[0].Length; j++)
                {
                    mW[i][j] = (mW[i][j] - ((1 - beta1) * gradient[i][j])) / beta1;
                }
            }

            // Revert second moment
            for (int i = 0; i < vW.Length; i++)
            {
                for (int j = 0; j < vW[0].Length; j++)
                {
                    vW[i][j] = (vW[i][j] - ((1 - beta2) * gradient[i][j] * gradient[i][j])) / beta2;
                }
            }
        }

        private void UpdateWeightWithAdam(string identifier, Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon, out HashSet<string> critical)
        {
            critical = new HashSet<string>();
            var example = new HashSet<string>();
            var avg = gradient[0].Select(x => Math.Abs(x)).Average();
            var max = gradient[0].Select(x => Math.Abs(x)).Max();
            var frobenius = w.FrobeniusNorm();
            avg = ((max - avg) / 8d) + avg;

            double scalingFactor = max > avg ? ((avg / max) / 1d) : 1d;
            this.scalingMap.TryAdd(w, scalingFactor);
            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    if (Math.Abs(gradient[i][j]) < avg)
                    {
                        gradient[i][j] = 0d; // *= scalingFactor;
                        critical.Add($"{i} {j}");
                    }
                    else
                    {
                        gradient[i][j] *= scalingFactor;
                        example.Add($"{i} {j}");
                    }
                }
            }

            /*
            if (this.SwitchGradients)
            {
                if (!this.e1Map.ContainsKey(w))
                {
                    this.e1Map.TryAdd(w, example);
                }
            }
            else
            {
                if (!this.e2Map.ContainsKey(w))
                {
                    this.e2Map.TryAdd(w, example);
                }
            }

            if (this.e1Map.ContainsKey(w) && this.e2Map.ContainsKey(w))
            {
                if (!this.intersectMap.ContainsKey(w))
                {
                    this.intersectMap.TryAdd(w, this.e1Map[w].Intersect(this.e2Map[w]).ToHashSet());
                }
            }
            */

            // Update biased first moment estimate
            var firstMoment = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta1, mW), MatrixUtils.ScalarMultiply(1 - beta1, gradient));

            // Update biased second raw moment estimate
            var secondMoment = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta2, vW), MatrixUtils.ScalarMultiply(1 - beta2, MatrixUtils.HadamardProduct(gradient, gradient)));

            // Compute bias-corrected first moment estimate
            Matrix mW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, this.network.Parameters.AdamIteration)), firstMoment);

            // Compute bias-corrected second raw moment estimate
            Matrix vW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, this.network.Parameters.AdamIteration)), secondMoment);

            // Update weights
            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    double weightReductionValue = this.network.Parameters.LearningRate * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);

                    // Compute new weight without actually updating it
                    double newWeight = w[i][j] - weightReductionValue;

                    // Check if the magnitude increases
                    if (frobenius > 0.0d && Math.Abs(newWeight) > Math.Abs(w[i][j]))
                    {
                        // Calculate the scaling factor based on the Frobenius norm
                        double sFactor = 1 / (1 + frobenius);

                        weightReductionValue *= sFactor; // Apply scaling directly to weightReductionValue
                    }

                    w[i][j] -= weightReductionValue;
                }
            }

            if (critical.Count == 0 && Math.Abs(gradient[0][0]) > 0d)
            {
            }

            this.matrixMap.TryAdd(w, critical);
            if (!this.scalingMap.ContainsKey(w))
            {
                bool res = this.scalingMap.TryAdd(w, scalingFactor);
            }

            // Update first moment
            for (int i = 0; i < mW.Length; i++)
            {
                for (int j = 0; j < mW[0].Length; j++)
                {
                    mW[i][j] = firstMoment[i][j];
                }
            }

            // Update second moment
            for (int i = 0; i < vW.Length; i++)
            {
                for (int j = 0; j < vW[0].Length; j++)
                {
                    vW[i][j] = secondMoment[i][j];
                }
            }
        }
    }
}
