//------------------------------------------------------------------------------
// <copyright file="HuberLossOperation.cs" author="ameritusweb" date="7/6/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    /// <summary>
    /// The Huber loss operation.
    /// </summary>
    public class HuberLossOperation : Operation
    {
        // Store the target matrix for use in the backward pass
        private Matrix target;

        // Store the delta value for the Huber loss
        private double delta;

        /// <summary>
        /// Initializes a new instance of the <see cref="HuberLossOperation"/> class.
        /// </summary>
        /// <param name="delta">The delta value for the Huber loss.</param>
        public HuberLossOperation(double delta = 1.0)
        {
            this.delta = delta;
        }

        /// <summary>
        /// A factory method for creating a Huber loss function.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated Huber loss operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new HuberLossOperation(net.Parameters.HuberLossDelta);
        }

        /// <summary>
        /// The forward pass of the Huber loss function.
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
            var loss = PradTools.Zero;
            var delta = (float)this.delta;

            // Using LINQ to calculate the Huber loss
            loss = (from row in Enumerable.Range(0, rowCount)
                    from col in Enumerable.Range(0, colCount)
                    let diff = output[row][col] - target[row][col]
                    select Math.Abs(diff) <= delta
                           ? PradTools.Half * diff * diff
                           : delta * (Math.Abs(diff) - (PradTools.Half * delta))).Sum();

            Matrix lossMatrix = new Matrix(1, 1);
            lossMatrix[0][0] = loss / (rowCount * colCount);
            return lossMatrix;
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
                    var diff = output[row][col] - this.target[row][col];
                    if (Math.Abs(diff) <= this.delta)
                    {
                        gradient[row][col] = scale * diff;
                    }
                    else
                    {
                        gradient[row][col] = scale * (float)this.delta * Math.Sign(diff);
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(gradient)
                .Build();
        }
    }
}
