//------------------------------------------------------------------------------
// <copyright file="VariedMaskedSoftmaxOperation.cs" author="ameritusweb" date="12/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Varied masked softmax operation.
    /// </summary>
    public class VariedMaskedSoftmaxOperation : Operation
    {
        private double temperature;
        private double maskThreshold;
        private Matrix input;
        private Matrix tempMatrix;
        private Matrix previousSoftmaxOutput;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new VariedMaskedSoftmaxOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrices.AddOrUpdate(id, this.input, (x, y) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the softmax function with temperature scaling and masking.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <param name="temp">The temperature to use for the softmax operation.</param>
        /// <param name="previousSoftmaxOutput">The previous softmax output used for masking.</param>
        /// <param name="maskThreshold">The threshold above which values in the previous output are masked.</param>
        /// <returns>The output of the softmax operation with masking.</returns>
        public Matrix Forward(Matrix input, Matrix temp, Matrix previousSoftmaxOutput, double maskThreshold)
        {
            this.input = input;
            this.tempMatrix = temp;
            this.previousSoftmaxOutput = previousSoftmaxOutput;
            this.maskThreshold = maskThreshold;

            // Compute the temperature sum
            this.temperature = temp.SelectMany(x => x).Sum();

            // Compute sumExp with masking applied
            double sumExp = 0;
            for (int i = 0; i < this.input.Cols; i++)
            {
                if (previousSoftmaxOutput[0][i] <= maskThreshold)
                {
                    sumExp += Math.Exp(this.input[0][i] / this.temperature);
                }
            }

            // Compute softmax values with masking
            double[] softmax = new double[this.input.Cols];
            for (int i = 0; i < this.input.Cols; i++)
            {
                if (previousSoftmaxOutput[0][i] <= maskThreshold)
                {
                    softmax[i] = Math.Exp(this.input[0][i] / this.temperature) / sumExp;
                }
                else
                {
                    softmax[i] = 0;
                }
            }

            // Apply scaling factor
            double scaleFactor = Math.Sqrt(this.input.Cols) / softmax.Sum();
            this.Output = new Matrix(softmax.Select(s => s * scaleFactor).ToArray());

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numCols = this.input.Cols;
            double[] softmax = new double[numCols];
            double sumExp = 0;

            // Recompute softmax with masking
            for (int i = 0; i < numCols; i++)
            {
                if (this.previousSoftmaxOutput[0][i] <= this.maskThreshold)
                {
                    double expVal = Math.Exp(this.input[0][i] / this.temperature);
                    softmax[i] = expVal;
                    sumExp += expVal;
                }
                else
                {
                    softmax[i] = 0;
                }
            }

            double softmaxSum = softmax.Sum();
            double scaleFactor = Math.Sqrt(numCols) / softmaxSum;
            double scaleFactorGradient = -Math.Sqrt(numCols) / Math.Pow(softmaxSum, 2);

            Matrix dX = new Matrix(1, numCols);
            Matrix dTemp = new Matrix(1, numCols);

            double[,] softmaxGrad = new double[numCols, numCols];
            double[,] gradSoftmaxTemp = new double[numCols, numCols];

            // Parallelize the outer loop
            Parallel.For(0, numCols, i =>
            {
                if (this.previousSoftmaxOutput[0][i] <= this.maskThreshold)
                {
                    for (int j = 0; j < numCols; j++)
                    {
                        softmaxGrad[i, j] = softmax[i] * ((i == j ? 1 : 0) - softmax[j]);
                        gradSoftmaxTemp[i, j] = -this.input[0][j] / Math.Pow(this.temperature, 2) * softmaxGrad[i, j];
                    }
                }
            });

            Parallel.For(0, numCols, i =>
            {
                if (this.previousSoftmaxOutput[0][i] <= this.maskThreshold)
                {
                    double gradientSumX = 0;
                    double gradientSumTemp = 0;

                    for (int j = 0; j < numCols; j++)
                    {
                        double totalGradX = ((softmaxGrad[i, j] / this.temperature) * scaleFactor * dLdOutput[0][j]) +
                                            (softmax[i] * scaleFactorGradient * dLdOutput[0][j]);
                        gradientSumX += totalGradX;

                        double totalGradTemp = (gradSoftmaxTemp[i, j] * scaleFactor * dLdOutput[0][j]) +
                                               (softmax[i] * scaleFactorGradient * dLdOutput[0][j]);
                        gradientSumTemp += totalGradTemp;
                    }

                    dX[0][i] = gradientSumX;
                    dTemp[0][i] = gradientSumTemp;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dX)
                .AddInputGradient(dTemp)
                .Build();
        }
    }
}
