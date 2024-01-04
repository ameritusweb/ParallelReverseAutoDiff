//------------------------------------------------------------------------------
// <copyright file="VariedSoftmaxOperation.cs" author="ameritusweb" date="12/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    /// <summary>
    /// Varied softmax operation.
    /// </summary>
    public class VariedSoftmaxOperation : Operation
    {
        private double temperature;
        private Matrix input;
        private Matrix tempMatrix;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new VariedSoftmaxOperation();
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
        /// Performs the forward operation for the softmax function.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <param name="temp">The temperature to use for the softmax operation.</param>
        /// <returns>The output of the softmax operation.</returns>
        public Matrix Forward(Matrix input, Matrix temp) // both input and temp are 1xN matrices
        {
            this.input = input;
            this.tempMatrix = temp;
            this.temperature = temp.SelectMany(x => x).Sum();
            double[] softmax = new double[this.input.Cols];
            double sumExp = this.input.SelectMany(x => x).Sum(xi => Math.Exp(xi / this.temperature));

            for (int i = 0; i < this.input.Cols; i++)
            {
                softmax[i] = Math.Exp(this.input[0][i] / this.temperature) / sumExp;
            }

            double scaleFactor = Math.Sqrt(this.input.Cols) / softmax.Sum();
            this.Output = new Matrix(softmax.Select(s => s * scaleFactor).ToArray()); // returns an 1xN matrix
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numCols = this.input.Cols;
            double sumExp = this.input.SelectMany(x => x).Sum(xi => Math.Exp(xi / this.temperature));
            double[] softmax = new double[numCols];
            for (int i = 0; i < numCols; i++)
            {
                softmax[i] = Math.Exp(this.input[0][i] / this.temperature) / sumExp;
            }

            double softmaxSum = softmax.Sum();
            double scaleFactor = Math.Sqrt(numCols) / softmaxSum;
            double scaleFactorGradient = -Math.Sqrt(numCols) / Math.Pow(softmaxSum, 2);
            Matrix dX = new Matrix(1, numCols);

            double[,] softmaxGrad = new double[numCols, numCols];

            Parallel.For(0, numCols, i =>
            {
                double gradientSum = 0;
                for (int j = 0; j < numCols; j++)
                {
                    softmaxGrad[i, j] = softmax[i] * ((i == j ? 1 : 0) - softmax[j]);
                    double totalGrad = ((softmaxGrad[i, j] / this.temperature) * scaleFactor * dLdOutput[0][j]) +
                                       (softmax[i] * scaleFactorGradient * dLdOutput[0][j]);
                    gradientSum += totalGrad;
                }

                dX[0][i] = gradientSum;
            });

            Matrix dTemp = new Matrix(1, this.tempMatrix.Cols);

            if (this.tempMatrix.Cols == numCols)
            {
                double[,] gradSoftmaxTemp = new double[1, numCols];

                for (int j = 0; j < numCols; j++)
                {
                    gradSoftmaxTemp[0, j] = -this.input[0][j] / Math.Pow(this.temperature, 2) * softmaxGrad[0, j];
                }

                double gradientSumTemp = 0;

                for (int j = 0; j < numCols; j++)
                {
                    double totalGradTemp = (gradSoftmaxTemp[0, j] * scaleFactor * dLdOutput[0][j]) +
                                            (softmax[0] * scaleFactorGradient * dLdOutput[0][j]);
                    gradientSumTemp += totalGradTemp;
                }

                dTemp[0][0] = gradientSumTemp;
                double scalar = gradientSumTemp / dX[0][0];

                for (int j = 1; j < numCols; ++j)
                {
                    dTemp[0][j] = scalar * dX[0][j];
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dX)
                .AddInputGradient(dTemp)
                .Build();
        }
    }
}
