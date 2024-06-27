//------------------------------------------------------------------------------
// <copyright file="ModifiedSoftmaxOperation.cs" author="ameritusweb" date="12/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// Modified softmax operation.
    /// </summary>
    public class ModifiedSoftmaxOperation : Operation
    {
        private const float Temperature = 0.1f;
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ModifiedSoftmaxOperation();
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
            float[] softmax = new float[this.input.Cols];
            float sumExp = this.input.SelectMany(x => x).Sum(xi => PradMath.Exp(xi / Temperature));

            for (int i = 0; i < this.input.Cols; i++)
            {
                softmax[i] = PradMath.Exp(this.input[0][i] / Temperature) / sumExp;
            }

            float scaleFactor = PradMath.Sqrt(this.input.Cols) / softmax.Sum();
            this.Output = new Matrix(softmax.Select(s => s * scaleFactor).ToArray()); // returns an 1xN matrix
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int numCols = this.input.Cols;
            float sumExp = this.input.SelectMany(x => x).Sum(xi => PradMath.Exp(xi / Temperature));
            float[] softmax = new float[numCols];
            for (int i = 0; i < numCols; i++)
            {
                softmax[i] = PradMath.Exp(this.input[0][i] / Temperature) / sumExp;
            }

            float softmaxSum = softmax.Sum();
            float scaleFactor = PradMath.Sqrt(numCols) / softmaxSum;
            float scaleFactorGradient = -PradMath.Sqrt(numCols) / PradMath.Pow(softmaxSum, 2);
            Matrix dX = new Matrix(1, numCols);

            for (int i = 0; i < numCols; i++)
            {
                float gradientSum = 0;
                for (int j = 0; j < numCols; j++)
                {
                    float softmaxGrad = softmax[i] * ((i == j ? 1 : 0) - softmax[j]);
                    float totalGrad = ((softmaxGrad / Temperature) * scaleFactor * dLdOutput[0][j]) +
                                       (softmax[i] * scaleFactorGradient * dLdOutput[0][j]);
                    gradientSum += totalGrad;
                }

                dX[0][i] = gradientSum;
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dX)
                .Build();
        }
    }
}
