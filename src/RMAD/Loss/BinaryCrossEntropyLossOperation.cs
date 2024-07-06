//------------------------------------------------------------------------------
// <copyright file="BinaryCrossEntropyLossOperation.cs" author="ameritusweb" date="7/6/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// The Binary cross entropy loss operation.
    /// </summary>
    public class BinaryCrossEntropyLossOperation : Operation
    {
        // Store the target matrix for use in the backward pass
        private Matrix target;

        /// <summary>
        /// A factory method for creating a binary cross entropy loss function.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated binary cross entropy operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new BinaryCrossEntropyLossOperation();
        }

        /// <summary>
        /// The forward pass of the binary cross entropy loss function.
        /// </summary>
        /// <param name="output">The output matrix.</param>
        /// <param name="target">The target matrix.</param>
        /// <returns>The loss matrix.</returns>
        public Matrix Forward(Matrix output, Matrix target)
        {
            this.target = target ?? throw new ArgumentNullException(nameof(target));
            if (output.Length != target.Length || output[0].Length != target[0].Length)
            {
                throw new ArgumentException("Output and target matrices must have the same dimensions.");
            }

            int rowCount = output.Length;
            int colCount = output[0].Length;
            var bce = PradTools.Zero;

            // Using LINQ to calculate the BCE
            bce = (from row in Enumerable.Range(0, rowCount)
                   from col in Enumerable.Range(0, colCount)
                   let y = target[row][col]
                   let yHat = output[row][col]
                   select (-y * PradMath.Log(yHat + PradTools.Epsilon)) - ((1 - y) * PradMath.Log(1 - yHat + PradTools.Epsilon))).Sum();

            Matrix loss = new Matrix(1, 1);
            loss[0][0] = bce / (rowCount * colCount);
            return loss;
        }

        /// <summary>
        /// Calculates the gradient of the loss function with respect to the output of the neural network.
        /// </summary>
        /// <param name="output">The output matrix.</param>
        /// <returns>The gradient.</returns>
        public override BackwardResult Backward(Matrix output)
        {
            int rowCount = output.Length;
            int colCount = output[0].Length;
            Matrix gradient = new Matrix(rowCount, colCount);

            var scale = PradTools.One / (rowCount * colCount);
            for (int row = 0; row < rowCount; row++)
            {
                for (int col = 0; col < colCount; col++)
                {
                    var y = this.target[row][col];
                    var yHat = output[row][col];
                    gradient[row][col] = scale * ((yHat - y) / ((yHat * (1 - yHat)) + PradTools.Epsilon));
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(gradient)
                .Build();
        }
    }
}
