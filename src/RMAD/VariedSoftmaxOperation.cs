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
            double[] expInputs = this.input.SelectMany(x => x).Select(xi => Math.Exp(xi / this.temperature)).ToArray();
            double sumExp = expInputs.Sum();

            double[] softmax = expInputs.Select(expInput => expInput / sumExp).ToArray();
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
                for (int j = 0; j < numCols; j++)
                {
                    softmaxGrad[i, j] = softmax[i] * ((i == j ? 1 : 0) - softmax[j]);
                    gradSoftmaxTemp[i, j] = -this.input[0][j] / Math.Pow(this.temperature, 2) * softmaxGrad[i, j];
                }
            });

            Parallel.For(0, numCols, i =>
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
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dX)
                .AddScalingGradient(dTemp)
                .Build();
        }
    }
}
