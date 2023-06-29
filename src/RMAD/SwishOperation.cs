//------------------------------------------------------------------------------
// <copyright file="SwishOperation.cs" author="ameritusweb" date="5/16/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Performs the forward and backward operations for the Swish activation function.
    /// </summary>
    public class SwishOperation : Operation
    {
        private Matrix input;
        private Matrix beta;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SwishOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.beta }, (key, oldValue) => new[] { this.input, this.beta });
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            var restored = this.IntermediateMatrixArrays[id];
            this.input = restored[0];
            this.beta = restored[1];
        }

        /// <summary>
        /// Performs the forward operation for the Swish activation function.
        /// </summary>
        /// <param name="input">The input to the Swish operation.</param>
        /// <param name="beta">The beta parameter.</param>
        /// <returns>The output of the Swish operation.</returns>
        public Matrix Forward(Matrix input, Matrix beta)
        {
            this.input = input;
            this.beta = beta;
            int rows = input.Rows;
            int cols = input.Cols;

            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = input[i, j];
                    double swish = this.Swish(x, beta);
                    this.Output[i, j] = swish;
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Rows;
            int cols = dLdOutput.Cols;
            Matrix dLdInput = new Matrix(rows, cols);
            Matrix dLdBeta = new Matrix(1, 1);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = this.input[i, j];
                    dLdInput[i, j] = dLdOutput[i, j] * this.DerivativeSwish(x, this.beta);

                    // Compute the gradient of loss with respect to beta
                    double sigmoid = 1 / (1 + Math.Exp(-this.beta[0][0] * x));
                    double sigmoidDerivative = sigmoid * (1 - sigmoid);
                    double swishDerivativeWithRespectToBeta = (x * sigmoidDerivative) - (Math.Pow(x, 2d) * Math.Pow(sigmoid, 2d));
                    dLdBeta[0][0] += dLdOutput[i, j] * swishDerivativeWithRespectToBeta;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .AddBetaGradient(dLdBeta)
                .Build();
        }

        private double Swish(double x, Matrix beta)
        {
            double sigmoid = 1 / (1 + Math.Exp(-beta[0][0] * x));
            return x * sigmoid;
        }

        private double DerivativeSwish(double x, Matrix beta)
        {
            double sigmoid = 1 / (1 + Math.Exp(-beta[0][0] * x));
            return sigmoid * ((1 + (beta[0][0] * x)) * (1 - sigmoid));
        }
    }
}
