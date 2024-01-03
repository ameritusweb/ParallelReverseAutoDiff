//------------------------------------------------------------------------------
// <copyright file="MultiRowModifiedSoftmaxOperation.cs" author="ameritusweb" date="12/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Multi-Row Modified softmax operation.
    /// </summary>
    public class MultiRowModifiedSoftmaxOperation : Operation
    {
        private const double Temperature = 0.1d;
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MultiRowModifiedSoftmaxOperation();
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
        /// <returns>The output of the softmax operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            var outputMatrix = new Matrix(input.Rows, input.Cols);

            for (int row = 0; row < input.Rows; row++)
            {
                double sumExp = this.input[row].Sum(xi => Math.Exp(xi / Temperature));
                double[] softmax = new double[input.Cols];

                for (int col = 0; col < input.Cols; col++)
                {
                    softmax[col] = Math.Exp(input[row, col] / Temperature) / sumExp;
                }

                double scaleFactor = Math.Sqrt(input.Cols) / softmax.Sum();
                for (int col = 0; col < input.Cols; col++)
                {
                    outputMatrix[row, col] = softmax[col] * scaleFactor;
                }
            }

            this.Output = outputMatrix;
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            Matrix dX = new Matrix(this.input.Rows, this.input.Cols);

            for (int row = 0; row < this.input.Rows; row++)
            {
                double sumExp = this.input[row].Sum(xi => Math.Exp(xi / Temperature));
                double[] softmax = new double[this.input.Cols];
                for (int i = 0; i < this.input.Cols; i++)
                {
                    softmax[i] = Math.Exp(this.input[row, i] / Temperature) / sumExp;
                }

                double softmaxSum = softmax.Sum();
                double scaleFactor = Math.Sqrt(this.input.Cols) / softmaxSum;
                double scaleFactorGradient = -Math.Sqrt(this.input.Cols) / Math.Pow(softmaxSum, 2);

                for (int i = 0; i < this.input.Cols; i++)
                {
                    double gradientSum = 0;
                    for (int j = 0; j < this.input.Cols; j++)
                    {
                        double softmaxGrad = softmax[i] * ((i == j ? 1 : 0) - softmax[j]);
                        double totalGrad = ((softmaxGrad / Temperature) * scaleFactor * dLdOutput[row, j]) +
                                           (softmax[i] * scaleFactorGradient * dLdOutput[row, j]);
                        gradientSum += totalGrad;
                    }

                    dX[row, i] = gradientSum;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dX)
                .Build();
        }
    }
}
