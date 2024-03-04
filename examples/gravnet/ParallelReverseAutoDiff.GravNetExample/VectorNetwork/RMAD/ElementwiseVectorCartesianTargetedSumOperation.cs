//------------------------------------------------------------------------------
// <copyright file="ElementwiseVectorCartesianTargetedSumOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using ParallelReverseAutoDiff.GravNetExample.GlyphNetwork;

    /// <summary>
    /// Element-wise cartesian sum operation.
    /// </summary>
    public class ElementwiseVectorCartesianTargetedSumOperation : Operation
    {
        private Matrix rotationTargets;
        private Matrix inputVectors;
        private int target;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ElementwiseVectorCartesianTargetedSumOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise vector rotation function.
        /// </summary>
        /// <param name="inputVectors">The first input to the element-wise vector rotation operation.</param>
        /// <param name="rotationTargets">The second input to the element-wise vector rotation operation.</param>
        /// <param name="target">The target index.</param>
        /// <returns>The output of the element-wise vector rotation operation.</returns>
        public Matrix Forward(Matrix inputVectors, Matrix rotationTargets, int target)
        {
            GlyphTrainingDynamics.Instance.PreviousTargetedSum[target] = GlyphTrainingDynamics.Instance.LastTargetedSum[target];
            GlyphTrainingDynamics.Instance.LastTargetedSum[target] = inputVectors;
            this.inputVectors = inputVectors;
            this.rotationTargets = rotationTargets;
            this.target = target;

            // Initialize the summation vector
            double sumX = 0.0;
            double sumY = 0.0;

            int vectorIndex = 0;
            for (int i = 0; i < 15; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    double x = inputVectors[vectorIndex, 0];
                    double y = inputVectors[vectorIndex, 1];

                    if (rotationTargets[i, j] == target)
                    {
                        // Summation
                        sumX += x;
                        sumY += y;
                    }

                    vectorIndex++;
                }
            }

            // Output is a 1x2 matrix of X and Y components for the output summation vector
            Matrix output = new Matrix(1, 2);
            output[0, 0] = sumX;
            output[0, 1] = sumY;
            this.Output = output;

            return this.Output;
        }


        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            // Initialize dInputVectors with the same shape as the forward input vectors
            Matrix dInputVectors = new Matrix(225, 2);
            var inputVectors = this.inputVectors;

            int vectorIndex = 0;
            for (int i = 0; i < 15; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    if (this.rotationTargets[i, j] == this.target)
                    {
                        dInputVectors[vectorIndex, 0] = dOutput[0, 0];
                        dInputVectors[vectorIndex, 1] = dOutput[0, 1];

                        var gradientDirection = GlyphTrainingDynamics.Instance.CalculateGradientDirection(inputVectors[vectorIndex, 0], inputVectors[vectorIndex, 1], this.target);
                        dInputVectors[vectorIndex, 0] = Math.Abs(dInputVectors[vectorIndex, 0]) * gradientDirection.Item1;
                        dInputVectors[vectorIndex, 1] = Math.Abs(dInputVectors[vectorIndex, 1]) * gradientDirection.Item2;
                    }

                    vectorIndex++;
                }
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInputVectors)
                .Build();
        }
    }
}
