//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorDotProductOperation.cs" author="ameritusweb" date="1/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise vector dot product operation.
    /// </summary>
    public class ElementwiseVectorDotProductOperation : Operation
    {
        private Matrix input;
        private Matrix weights;
        private CalculatedValues[] calculatedValues;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorDotProductOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector dot product function.
        /// </summary>
        /// <param name="input">The first input to the element-wise vector dot product operation.</param>
        /// <param name="weights">The second input to the element-wise vector dot product operation.</param>
        /// <returns>The output of the element-wise vector dot product operation.</returns>
        public Matrix Forward(Matrix input, Matrix weights)
        {
            // Validate dimensions
            if (input.Rows != weights.Rows || input.Cols != 2 || weights.Cols != 2)
            {
                throw new ArgumentException("Input and weights matrices must have the same number of rows and 2 columns each.");
            }

            this.input = input;
            this.weights = weights;
            this.Output = new Matrix(input.Rows, 1);

            this.calculatedValues = new CalculatedValues[input.Rows];

            Parallel.For(0, input.Rows, i =>
            {
                var magnitude = weights[i, 0];
                var angle = weights[i, 1];
                double x1 = magnitude * Math.Cos(angle);
                double y1 = magnitude * Math.Sin(angle);
                double dotProduct = input[i, 0] * x1 + input[i, 1] * y1;

                // Derivative of dot product with respect to magnitude
                double dDotProduct_dMagnitude = input[i, 0] * Math.Cos(angle) + input[i, 1] * Math.Sin(angle);
                this.calculatedValues[i].dDotProduct_dMagnitude = dDotProduct_dMagnitude;

                double dDotProduct_dInputMagnitude = magnitude * Math.Cos(angle);
                this.calculatedValues[i].dDotProduct_dInputMagnitude = dDotProduct_dInputMagnitude;

                // Derivative of dot product with respect to angle
                double dDotProduct_dAngle = input[i, 0] * (-magnitude * Math.Sin(angle)) + input[i, 1] * (magnitude * Math.Cos(angle));
                this.calculatedValues[i].dDotProduct_dAngle = dDotProduct_dAngle;

                double dDotProduct_dInputAngle = magnitude * Math.Sin(angle);
                this.calculatedValues[i].dDotProduct_dInputAngle = dDotProduct_dInputAngle;

                this.Output[i, 0] = dotProduct;
            });

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput = new Matrix(this.input.Rows, 2);
            Matrix dWeights = new Matrix(this.weights.Rows, 2);

            Parallel.For(0, this.input.Rows, i =>
            {
                dInput[i, 0] = dOutput[i, 0] * this.calculatedValues[i].dDotProduct_dInputMagnitude;
                dInput[i, 1] = dOutput[i, 0] * this.calculatedValues[i].dDotProduct_dInputAngle;

                dWeights[i, 0] = dOutput[i, 0] * this.calculatedValues[i].dDotProduct_dMagnitude;
                dWeights[i, 1] = dOutput[i, 0] * this.calculatedValues[i].dDotProduct_dAngle;
            });

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddInputGradient(dWeights)
                .Build();
        }

        private struct CalculatedValues
        {
            public double dDotProduct_dMagnitude { get; set; }

            public double dDotProduct_dAngle { get; set; }

            public double dDotProduct_dInputMagnitude { get; set; }

            public double dDotProduct_dInputAngle { get; set; }
        }
    }
}
