//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorAddOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise add operation.
    /// </summary>
    public class ElementwiseVectorAddOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorAddOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector summation function.
        /// </summary>
        /// <param name="input1">The first input to the element-wise vector summation operation.</param>
        /// <param name="input2">The second input to the element-wise vector summation operation.</param>
        /// <returns>The output of the element-wise vector summation operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2)
        {
            this.input1 = input1;
            this.input2 = input2;

            this.Output = new Matrix(this.input1.Rows, this.input1.Cols);
            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    var magnitude = input1[i, j];
                    var angle = input1[i, j + (input1.Cols / 2)];

                    var wMagnitude = input2[i, j];
                    var wAngle = input2[i, j + (input2.Cols / 2)];

                    // Compute vector components
                    var x1 = magnitude * PradMath.Cos(angle);
                    var y1 = magnitude * PradMath.Sin(angle);
                    var x2 = wMagnitude * PradMath.Cos(wAngle);
                    var y2 = wMagnitude * PradMath.Sin(wAngle);

                    var sumx = x1 + x2;
                    var sumy = y1 + y2;

                    // Compute resultant vector magnitude and angle
                    var resultMagnitude = PradMath.Sqrt((sumx * sumx) + (sumy * sumy));
                    var resultAngle = PradMath.Atan2(sumy, sumx);

                    this.Output[i, j] = resultMagnitude;
                    this.Output[i, j + (this.input1.Cols / 2)] = resultAngle;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);

            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    var magnitude = this.input1[i, j];
                    var angle = this.input1[i, j + (this.input1.Cols / 2)];
                    var wMagnitude = this.input2[i, j];
                    var wAngle = this.input2[i, j + (this.input2.Cols / 2)];

                    var x1 = magnitude * PradMath.Cos(angle);
                    var y1 = magnitude * PradMath.Sin(angle);
                    var x2 = wMagnitude * PradMath.Cos(wAngle);
                    var y2 = wMagnitude * PradMath.Sin(wAngle);

                    var combinedX = x1 + x2;
                    var combinedY = y1 + y2;

                    // Compute gradients for magnitude and angle
                    var dResultMagnitude_dX = combinedX / this.Output[i, j];
                    var dResultMagnitude_dY = combinedY / this.Output[i, j];

                    var dResultAngle_dX = -combinedY / ((combinedX * combinedX) + (combinedY * combinedY));
                    var dResultAngle_dY = combinedX / ((combinedX * combinedX) + (combinedY * combinedY));

                    // Chain rule to compute gradients for input vectors
                    dInput1[i, j] = (dOutput[i, j] * dResultMagnitude_dX * PradMath.Cos(angle)) +
                                    (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dX * -PradMath.Sin(angle));
                    dInput1[i, j + (this.input1.Cols / 2)] = (dOutput[i, j] * dResultMagnitude_dY * PradMath.Sin(angle)) +
                                                             (dOutput[i, j + (this.input1.Cols / 2)] * dResultAngle_dY * PradMath.Cos(angle));

                    dInput2[i, j] = (dOutput[i, j] * dResultMagnitude_dX * PradMath.Cos(wAngle)) +
                                    (dOutput[i, j + (this.input2.Cols / 2)] * dResultAngle_dX * -PradMath.Sin(wAngle));
                    dInput2[i, j + (this.input2.Cols / 2)] = (dOutput[i, j] * dResultMagnitude_dY * PradMath.Sin(wAngle)) +
                                                             (dOutput[i, j + (this.input2.Cols / 2)] * dResultAngle_dY * PradMath.Cos(wAngle));
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .Build();
        }
    }
}
