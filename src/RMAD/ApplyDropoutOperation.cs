//------------------------------------------------------------------------------
// <copyright file="ApplyDropoutOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Applies dropout to a small portion of the input.
    /// </summary>
    public class ApplyDropoutOperation : Operation
    {
        private readonly double dropoutRate;
        private readonly Random random;
        private Matrix input;
        private Matrix dropoutMask;

        /// <summary>
        /// Initializes a new instance of the <see cref="ApplyDropoutOperation"/> class.
        /// </summary>
        /// <param name="dropoutRate">The dropout rate to apply.</param>
        public ApplyDropoutOperation(double dropoutRate)
        {
            this.dropoutRate = dropoutRate;
            this.random = new Random(Guid.NewGuid().GetHashCode());
        }

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ApplyDropoutOperation(net.Parameters.DropoutRate);
        }

        /// <summary>
        /// The forward pass of the apply dropout operation.
        /// </summary>
        /// <param name="input">The input for the apply dropout operation.</param>
        /// <returns>The output for the apply dropout operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            this.Output = new Matrix(numRows, numCols);
            this.dropoutMask = new Matrix(numRows, numCols);

            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    double randomValue = this.random.NextDouble();
                    this.dropoutMask[i][j] = randomValue < this.dropoutRate ? 0 : 1;
                    this.Output[i][j] = this.dropoutMask[i][j] * input[i][j];
                }
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override (Matrix?, Matrix?) Backward(Matrix dLdOutput)
        {
            int numRows = dLdOutput.Length;
            int numCols = dLdOutput[0].Length;

            Matrix dLdInput = new Matrix(numRows, numCols);
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numCols; j++)
                {
                    dLdInput[i][j] = dLdOutput[i][j] * this.dropoutMask[i][j];
                }
            }

            return (dLdInput, dLdInput);
        }
    }
}
