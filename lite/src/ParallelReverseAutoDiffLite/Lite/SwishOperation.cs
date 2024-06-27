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
            this.IntermediateMatrixArrays.AddOrUpdate(id, new[] { this.input, this.beta }, (x, y) => new[] { this.input, this.beta });
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
                    float x = input[i, j];
                    float swish = this.Swish(x, beta);
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
                    float x = this.input[i, j];
                    dLdInput[i, j] = dLdOutput[i, j] * this.DerivativeSwish(x, this.beta);

                    // Compute the gradient of loss with respect to beta
                    float sigmoid = 1 / (1 + PradMath.Exp(-this.beta[0][0] * x));
                    float sigmoidDerivative = sigmoid * (1 - sigmoid);
                    float swishDerivativeWithRespectToBeta = (x * sigmoidDerivative) - (PradMath.Pow(x, 2f) * PradMath.Pow(sigmoid, 2f));
                    dLdBeta[0][0] += dLdOutput[i, j] * swishDerivativeWithRespectToBeta;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .AddBetaGradient(dLdBeta)
                .Build();
        }

        private float Swish(float x, Matrix beta)
        {
            float sigmoid = 1 / (1 + PradMath.Exp(-beta[0][0] * x));
            return x * sigmoid;
        }

        private float DerivativeSwish(float x, Matrix beta)
        {
            float sigmoid = 1 / (1 + PradMath.Exp(-beta[0][0] * x));
            return sigmoid * ((1 + (beta[0][0] * x)) * (1 - sigmoid));
        }
    }
}
