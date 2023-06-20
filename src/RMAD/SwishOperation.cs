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
            this.IntermediateMatrices.AddOrUpdate(id, this.input, (key, oldValue) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            this.input = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation for the Swish activation function.
        /// </summary>
        /// <param name="input">The input to the Swish operation.</param>
        /// <returns>The output of the Swish operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int rows = input.Rows;
            int cols = input.Cols;

            this.Output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = input[i, j];
                    double swish = this.Swish(x);
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

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = this.input[i, j];
                    dLdInput[i, j] = dLdOutput[i, j] * this.DerivativeSwish(x);
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }

        private double Swish(double x)
        {
            double sigmoid = 1 / (1 + Math.Exp(-x));
            return x * sigmoid;
        }

        private double DerivativeSwish(double x)
        {
            double sigmoid = 1 / (1 + Math.Exp(-x));
            return sigmoid + (x * sigmoid * (1 - sigmoid));
        }
    }
}
