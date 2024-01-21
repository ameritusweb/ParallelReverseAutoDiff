//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorPolarFftOperation.cs" author="ameritusweb" date="1/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector polar FFT operation.
    /// </summary>
    public class ElementwiseVectorPolarFftOperation : Operation
    {
        private Matrix input1;
        private Matrix input2;
        private Matrix weights;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorPolarFftOperation();
        }

        /// <summary>
        /// Performs the forward operation for the Hadamard product function.
        /// </summary>
        /// <param name="input1">The first input to the Hadamard product operation.</param>
        /// <param name="input2">The second input to the Hadamard product operation.</param>
        /// <returns>The output of the Hadamard product operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            this.input1 = input1;
            this.input2 = input2;
            this.weights = weights;

            this.Output = new Matrix(input1.Rows,input1.Cols);

            Parallel.For(0, input1.Rows, i =>
            {
                for (int j = 0; j < input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    double magnitude = input1[i, j];
                    double angle = input1[i, j + (input1.Cols / 2)];

                    double wMagnitude = input2[i, j];
                    double wAngle = input2[i, j + (input1.Cols / 2)];

                    this.Output[i, j] = magnitude * wMagnitude * weights[i, j];
                    this.Output[i, j + (input1.Cols / 2)] = angle + wAngle;
                }
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Initialize gradient matrices
            Matrix dInput1 = new Matrix(this.input1.Rows, this.input1.Cols);
            Matrix dInput2 = new Matrix(this.input2.Rows, this.input2.Cols);
            Matrix dWeights = new Matrix(this.weights.Rows, this.weights.Cols);

            // Loop through each element in input1
            Parallel.For(0, this.input1.Rows, i =>
            {
                for (int j = 0; j < this.input1.Cols / 2; j++)
                {
                    // Accessing the magnitudes and angles from the concatenated matrices
                    double magnitude = this.input1[i, j];
                    double angle = this.input1[i, j + (this.input1.Cols / 2)];

                    double wMagnitude = this.input2[i, j];
                    double wAngle = this.input2[i, j + (this.input1.Cols / 2)];

                    double dOutputMagnitude_dMagnitude = wMagnitude * this.weights[i, j];
                    double dOutputMagnitude_dwMagnitude = magnitude * this.weights[i, j];
                    double dOutputMagnitude_dWeight = magnitude * wMagnitude;
                    double dOutputAngle_dAngle = 1;
                    double dOutputAngle_dwAngle = 1;

                    // Applying the chain rule to calculate gradients
                    // Gradient of magnitude part of input1
                    dInput1[i, j] += dOutput[i, j] * dOutputMagnitude_dMagnitude;

                    // Gradient of angle part of input1
                    dInput1[i, j + (this.input1.Cols / 2)] += dOutput[i, j + (this.input1.Cols / 2)] * dOutputAngle_dAngle;

                    // Gradient of magnitude part of input2
                    dInput2[i, j] += dOutput[i, j] * dOutputMagnitude_dwMagnitude;

                    // Gradient of angle part of input2
                    dInput2[i, j + (this.input1.Cols / 2)] += dOutput[i, j + (this.input1.Cols / 2)] * dOutputAngle_dwAngle;

                    // Gradient of the weight
                    dWeights[i, j] += dOutput[i, j] * dOutputMagnitude_dWeight;
                }
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }
    }
}
