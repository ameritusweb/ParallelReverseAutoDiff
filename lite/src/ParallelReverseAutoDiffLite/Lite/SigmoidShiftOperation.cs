//------------------------------------------------------------------------------
// <copyright file="SigmoidShiftOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Sigmoid shift operation.
    /// </summary>
    public class SigmoidShiftOperation : Operation
    {
        private Matrix input;
        private float minValue;
        private (int row, int col) minPosition;
        private float sigmoidMin;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SigmoidShiftOperation();
        }

        /// <summary>
        /// Performs the forward operation for the sigmoid shift function.
        /// </summary>
        /// <param name="input">The input to the sigmoid shift operation.</param>
        /// <returns>The output of the sigmoid shift operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            (this.minValue, this.minPosition) = this.FindMinimumWithValueAndPosition(input);

            // Step 2: Apply the sigmoid function to the minimum value
            this.sigmoidMin = this.Sigmoid(this.minValue);

            this.Output = this.ShiftMatrixValues(input);

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput = new Matrix(this.input.Rows, this.input.Cols);

            // Gradient of the sigmoid function wrt minimum value
            float dSigmoidMin = this.sigmoidMin * (1 - this.sigmoidMin); // Derivative of sigmoid

            // Update gradients
            for (int i = 0; i < dOutput.Rows; i++)
            {
                for (int j = 0; j < dOutput.Cols; j++)
                {
                    if (i == this.minPosition.row && j == this.minPosition.col)
                    {
                        // For the minimum element: combine gradients from subtraction and sigmoid operations
                        dInput[i, j] = dOutput[i, j] * dSigmoidMin;
                    }
                    else
                    {
                        // For other elements: pass the gradient through unchanged
                        dInput[i, j] = dOutput[i, j];
                    }
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }

        private (float minValue, (int row, int col) position) FindMinimumWithValueAndPosition(Matrix matrix)
        {
            float minValue = float.MaxValue;
            (int row, int col) position = (0, 0);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    if (matrix[i, j] < minValue)
                    {
                        minValue = matrix[i, j];
                        position = (i, j);
                    }
                }
            }

            return (minValue, position);
        }

        private Matrix ShiftMatrixValues(Matrix matrix)
        {
            Matrix result = new Matrix(matrix.Rows, matrix.Cols);
            Parallel.For(0, matrix.Rows, i =>
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    result[i, j] = matrix[i, j] - this.minValue + this.sigmoidMin;
                }
            });

            return result;
        }

        private float Sigmoid(float x)
        {
            return 1.0f / (1.0f + PradMath.Exp(-x));
        }
    }
}
