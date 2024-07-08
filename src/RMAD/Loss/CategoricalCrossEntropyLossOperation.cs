//------------------------------------------------------------------------------
// <copyright file="CategoricalCrossEntropyLossOperation.cs" author="ameritusweb" date="7/6/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD.LossOps
{
    using System;
    using System.Linq;
    using ParallelReverseAutoDiff.PRAD;

    /// <summary>
    /// The Categorical cross-entropy loss operation.
    /// </summary>
    public class CategoricalCrossEntropyLossOperation : PradOperationBase<BinaryCrossEntropyLossOperation, Matrix, Matrix, Matrix>
    {
        // Store the target matrix for use in the backward pass
        private Matrix target;

        /// <summary>
        /// A factory method for creating a categorical cross entropy loss function.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated categorical cross entropy operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new CategoricalCrossEntropyLossOperation();
        }

        /// <summary>
        /// The forward pass of the categorical cross entropy loss function.
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
            var cce = PradTools.Zero;

            // Using LINQ to calculate the CCE
            cce = (from row in Enumerable.Range(0, rowCount)
                   from col in Enumerable.Range(0, colCount)
                   let y = target[row][col]
                   let yHat = output[row][col]
                   select -y * PradMath.Log(yHat + PradTools.Epsilon)).Sum();

            Matrix loss = new Matrix(1, 1);
            loss[0][0] = cce / rowCount;
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

            var scale = PradTools.One / rowCount;
            for (int row = 0; row < rowCount; row++)
            {
                for (int col = 0; col < colCount; col++)
                {
                    var y = this.target[row][col];
                    var yHat = output[row][col];
                    gradient[row][col] = -scale * y / (yHat + PradTools.Epsilon);
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(gradient)
                .Build();
        }
    }
}
