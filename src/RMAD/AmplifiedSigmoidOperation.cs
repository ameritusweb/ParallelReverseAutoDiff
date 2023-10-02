//------------------------------------------------------------------------------
// <copyright file="AmplifiedSigmoidOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// The sigmoid operation utilizing gradient amplification.
    /// </summary>
    public class AmplifiedSigmoidOperation : Operation
    {
        private Matrix input;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new AmplifiedSigmoidOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            this.IntermediateMatrices.AddOrUpdate(id, this.input, (_, _) => this.input);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
           this.input = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// The forward pass of the operation.
        /// </summary>
        /// <param name="input">The input for the operation.</param>
        /// <returns>The output for the operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            this.Output = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    this.Output[i][j] = 1.0 / (1.0 + Math.Pow(Math.PI - 2, -input[i][j]));
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;
            Matrix dLdInput = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    double x = this.input[i][j];
                    double dx = Math.Pow(Math.PI - 2, -x) / Math.Pow(1 + Math.Pow(Math.PI - 2, -x), 2);
                    dLdInput[i][j] = dOutput[i][j] * dx;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
