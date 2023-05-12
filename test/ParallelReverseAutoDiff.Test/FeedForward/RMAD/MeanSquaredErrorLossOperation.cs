using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.FeedForward.RMAD
{
    /// <summary>
    /// The mean squared error loss operation.
    /// </summary>
    public class MeanSquaredErrorLossOperation : Operation
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
        public Matrix Forward(Matrix output, Matrix target)
        {
            this.target = target;
            int rowCount = output.Length;
            int colCount = output[0].Length;
            double mse = 0;

            for (int row = 0; row < rowCount; row++)
            {
                for (int col = 0; col < colCount; col++)
                {
                    double diff = output[row][col] - target[row][col];
                    mse += diff * diff;
                }
            }

            Matrix loss = new Matrix(1, 1);
            loss[0][0] = mse / (rowCount * colCount);
            return loss;
        }

        /// <summary>
        /// Calculates the gradient of the loss function with respect to the output of the neural network.
        /// </summary>
        /// <param name="output">The output matrix.</param>
        /// <returns>The gradient.</returns>
        public override (Matrix?, Matrix?) Backward(Matrix output)
        {
            int rowCount = output.Length;
            int colCount = output[0].Length;
            Matrix gradient = new Matrix(rowCount, colCount);

            double scale = 2.0 / (rowCount * colCount);
            for (int row = 0; row < rowCount; row++)
            {
                for (int col = 0; col < colCount; col++)
                {
                    gradient[row][col] = scale * (output[row][col] - this.target[row][col]);
                }
            }

            return (gradient, null);
        }
    }
}
