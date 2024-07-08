//------------------------------------------------------------------------------
// <copyright file="MeanSquaredErrorLossOperation.cs" author="ameritusweb" date="7/6/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD.LossOps
{
    using System;
    using System.Linq;
    using ParallelReverseAutoDiff.PRAD;

    /// <summary>
    /// The Mean squared error loss operation.
    /// </summary>
    public class MeanSquaredErrorLossOperation : PradOperationBase<BinaryCrossEntropyLossOperation, Matrix, Matrix, Matrix>
    {
        // Store the target matrix for use in the backward pass
        private Matrix target;

        /// <summary>
        /// A factory method for creating a mean squared error loss function.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated mean squared error operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MeanSquaredErrorLossOperation();
        }

        /// <summary>
        /// The forward pass of the mean squared error loss function.
        /// </summary>
        /// <param name="output">The output matrix.</param>
        /// <param name="target">The target matrix.</param>
        /// <returns>The loss matrix.</returns>
        public override Matrix Forward(Matrix output, Matrix target)
        {
            this.target = target ?? throw new ArgumentNullException(nameof(target));
            if (output.Length != target.Length || output[0].Length != target[0].Length)
            {
                throw new ArgumentException("Output and target matrices must have the same dimensions.");
            }

            int rowCount = output.Length;
            int colCount = output[0].Length;
            var mse = PradTools.Zero;

            // Using LINQ to calculate the MSE
            mse = (from row in Enumerable.Range(0, rowCount)
                   from col in Enumerable.Range(0, colCount)
                   let diff = output[row][col] - target[row][col]
                   select diff * diff).Sum();

            Matrix loss = new Matrix(1, 1);
            loss[0][0] = mse / (rowCount * colCount);
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

            var scale = PradTools.Two / (rowCount * colCount);
            for (int row = 0; row < rowCount; row++)
            {
                for (int col = 0; col < colCount; col++)
                {
                    gradient[row][col] = scale * (output[row][col] - this.target[row][col]);
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(gradient)
                .Build();
        }
    }
}
