//------------------------------------------------------------------------------
// <copyright file="StochasticAdamOptimizer.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition;
    using System;
    using System.Diagnostics;
    using System.Threading.Tasks;

    /// <summary>
    /// An Adam optimizer.
    /// </summary>
    public class StochasticAdamOptimizer
    {
        private readonly NeuralNetwork network;
        private readonly RandomNumberGenerator randGen = new RandomNumberGenerator();

        /// <summary>
        /// Initializes a new instance of the <see cref="AdamOptimizer"/> class.
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
                                this.UpdateWeightWithAdam(weightMatrix, firstMomentMatrix, secondMomentMatrix, gradientMatrix, this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon);
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
                                    this.UpdateWeightWithAdam(weightMatrix[d], firstMomentMatrix[d], secondMomentMatrix[d], gradientMatrix[d], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon);
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
                                        this.UpdateWeightWithAdam(weightMatrix[d1][d2], firstMomentMatrix[d1][d2], secondMomentMatrix[d1][d2], gradientMatrix[d1][d2], this.network.Parameters.AdamBeta1, this.network.Parameters.AdamBeta2, this.network.Parameters.AdamEpsilon);
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

        private void UpdateWeightWithAdam(Matrix w, Matrix mW, Matrix vW, Matrix gradient, double beta1, double beta2, double epsilon)
        {
            // Update biased first moment estimate
            var firstMoment = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta1, mW), MatrixUtils.ScalarMultiply(1 - beta1, gradient));

            // Update biased second raw moment estimate
            var secondMoment = MatrixUtils.MatrixAdd(MatrixUtils.ScalarMultiply(beta2, vW), MatrixUtils.ScalarMultiply(1 - beta2, MatrixUtils.HadamardProduct(gradient, gradient)));

            // Compute bias-corrected first moment estimate
            Matrix mW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta1, this.network.Parameters.AdamIteration)), firstMoment);

            // Compute bias-corrected second raw moment estimate
            Matrix vW_hat = MatrixUtils.ScalarMultiply(1 / (1 - Math.Pow(beta2, this.network.Parameters.AdamIteration)), secondMoment);

            List<double> randomRate = new List<double>();
            if (w[0][0] > 4d)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    randomRate.Add(randGen.GetRandomNumber(0.1d, 1d));
                }
            }
            else
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    randomRate.Add(randGen.GetRandomNumber(0.00001d, 0.001d));
                }
            }

            // Update weights
            for (int i = 0; i < w.Length; i++)
            {
                for (int j = 0; j < w[0].Length; j++)
                {
                    double weightReductionValue = this.network.Parameters.LearningRate * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);

                    if (weightReductionValue > 0d)
                    {
                        double ww = w[i][j];
                        weightReductionValue = randomRate[j] * mW_hat[i][j] / (Math.Sqrt(vW_hat[i][j]) + epsilon);
                    }

                    w[i][j] -= weightReductionValue;
                }
            }

            var aaaa = 1;

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
