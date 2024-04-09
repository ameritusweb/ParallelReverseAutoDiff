//------------------------------------------------------------------------------
// <copyright file="StochasticAdamOptimizer.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A stochastic Adam optimizer.
    /// </summary>
    public class StochasticAdamOptimizer
    {
        private readonly NeuralNetwork network;
        private readonly RandomNumberGenerator randGen = new RandomNumberGenerator();
        private Dictionary<string, int> counts = new Dictionary<string, int>();
        private Dictionary<string, (int, double)> countValues = new Dictionary<string, (int, double)>();

        /// <summary>
        /// Initializes a new instance of the <see cref="StochasticAdamOptimizer"/> class.
        /// </summary>
        /// <param name="network">The neural network.</param>
        public StochasticAdamOptimizer(NeuralNetwork network)
        {
            this.network = network;
        }

        /// <summary>
        /// Optimize the layers.
        /// </summary>
        /// <param name="layers">The layers to optimize.</param>
        public void Optimize(IModelLayer[] layers)
        {
            for (int i = 0; i < layers.Length; i++)
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
                                double frobeniusBefore = weightMatrix.FrobeniusNorm();
                                Dictionary<double, (int, int)> positiveChanges = new Dictionary<double, (int, int)>();
                                Dictionary<double, (int, int)> negativeChanges = new Dictionary<double, (int, int)>();
                                this.UpdateWeightWithAdam(identifier, weightMatrix, firstMomentMatrix, secondMomentMatrix, gradientMatrix, this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon, ref positiveChanges, ref negativeChanges);

                                // double frobeniusAfter = weightMatrix.FrobeniusNorm();
                                // double newFrobenius = 0.0d;
                                // int index = 0;
                                // if (identifier != "Queries" && identifier != "Keys")
                                // {
                                //    if (frobeniusAfter > frobeniusBefore)
                                //    {
                                //        var list1 = positiveChanges.OrderBy(x => x.Key).ToList();
                                //        var list2 = negativeChanges.OrderByDescending(x => x.Key).ToList();
                                //        var list3 = positiveChanges.OrderByDescending(x => x.Key).ToList();
                                //        var list4 = negativeChanges.OrderBy(x => x.Key).ToList();
                                //        do
                                //        {
                                //            for (int k = index; k < index + 100; ++k)
                                //            {
                                //                var (key1, (i1, j1)) = list1[k];
                                //                var (key2, (i2, j2)) = list2[k];
                                //                weightMatrix[i1][j1] += key1;
                                //                weightMatrix[i2][j2] += key2;
                                //            }
                                //            for (int k = index; k < index + 100; ++k)
                                //            {
                                //                var (key1, (i1, j1)) = list3[k];
                                //                var (key2, (i2, j2)) = list4[k];
                                //                weightMatrix[i1][j1] -= key1;
                                //                weightMatrix[i2][j2] -= key2;
                                //            }
                                //            newFrobenius = weightMatrix.FrobeniusNorm();
                                //            index += 100;
                                //            if ((index + 100) >= Math.Min(positiveChanges.Count, negativeChanges.Count))
                                //            {
                                //                break;
                                //            }
                                //        }
                                //        while (newFrobenius > frobeniusBefore);
                                //    }
                                // }
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
                                    Dictionary<double, (int, int)> positiveChanges = new Dictionary<double, (int, int)>();
                                    Dictionary<double, (int, int)> negativeChanges = new Dictionary<double, (int, int)>();
                                    this.UpdateWeightWithAdam(identifier, weightMatrix[d], firstMomentMatrix[d], secondMomentMatrix[d], gradientMatrix[d], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon, ref positiveChanges, ref negativeChanges);
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
                                        Dictionary<double, (int, int)> positiveChanges = new Dictionary<double, (int, int)>();
                                        Dictionary<double, (int, int)> negativeChanges = new Dictionary<double, (int, int)>();
                                        this.UpdateWeightWithAdam(identifier, weightMatrix[d1][d2], firstMomentMatrix[d1][d2], secondMomentMatrix[d1][d2], gradientMatrix[d1][d2], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon, ref positiveChanges, ref negativeChanges);
                                    }
                                }

                                break;
                            }
                    }
                }
            }
        }

        private void UpdateWeightWithAdam(string identifier, Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon, ref Dictionary<double, (int I, int J)> positiveChanges, ref Dictionary<double, (int I, int J)> negativeChanges)
        {
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

                    string key = $"{identifier} {i} {j}";
                    if (w[i][j] > 0.0d)
                    {
                        if (!positiveChanges.ContainsKey(weightReductionValue))
                        {
                            positiveChanges.Add(weightReductionValue, (i, j));
                        }
                    }
                    else if (w[i][j] < 0.0d)
                    {
                        if (!negativeChanges.ContainsKey(weightReductionValue))
                        {
                            negativeChanges.Add(weightReductionValue, (i, j));
                        }
                    }

                    if (weightReductionValue > 0.0d)
                    {
                        if (this.counts.ContainsKey(key))
                        {
                            var count = this.counts[key];
                            if (count < 0)
                            {
                                this.counts[key] = 0;
                            }
                            else
                            {
                                this.counts[key] = count + 1;
                            }
                        }
                        else
                        {
                            this.counts.Add(key, 1);
                        }
                    }
                    else if (weightReductionValue < 0.0d)
                    {
                        if (this.counts.ContainsKey(key))
                        {
                            var count = this.counts[key];
                            if (count > 0)
                            {
                                this.counts[key] = 0;
                            }
                            else
                            {
                                this.counts[key] = count - 1;
                            }
                        }
                        else
                        {
                            this.counts.Add(key, -1);
                        }
                    }

                    if (double.IsNaN(weightReductionValue))
                    {
                        Console.WriteLine("Weight reduction value is NaN.");
                    }

                    if (this.counts.ContainsKey(key) && (this.counts[key] >= 2 || this.counts[key] <= -2))
                    {
                        if (!this.countValues.ContainsKey(key))
                        {
                            this.countValues.Add(key, (this.counts[key], w[i][j]));
                        }
                        else
                        {
                            this.countValues[key] = (this.counts[key], w[i][j]);
                        }

                        var value = this.counts[key];
                        var absValue = Math.Min(75, Math.Abs(value));
                        var decay = Math.Pow(0.999d, absValue);
                        var newValue = weightReductionValue * absValue * decay;
                        if (Math.Abs(newValue) < this.network.Parameters.LearningRate)
                        {
                            weightReductionValue = newValue;
                        }
                        else
                        {
                        }
                    }

                    w[i][j] -= weightReductionValue;
                }
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
