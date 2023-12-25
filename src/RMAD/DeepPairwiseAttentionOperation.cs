//------------------------------------------------------------------------------
// <copyright file="DeepPairwiseAttentionOperation.cs" author="ameritusweb" date="12/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// A deep pairwise attention operation.
    /// </summary>
    public class DeepPairwiseAttentionOperation : DeepOperation
    {
        private DeepMatrix concatenatedMatrices; // Array of Nx(2M) matrices
        private Matrix sharedMatrix; // Shared 1x(2M) matrix

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IDeepOperation Instantiate(NeuralNetwork net)
        {
            return new DeepPairwiseAttentionOperation();
        }

        /// <summary>
        /// The forward pass of the deep pairwise attention operation.
        /// </summary>
        /// <param name="concatenatedMatrices">The input matrices.</param>
        /// <param name="sharedMatrix">The shared matrix.</param>
        /// <returns>The masked input.</returns>
        public DeepMatrix Forward(DeepMatrix concatenatedMatrices, Matrix sharedMatrix)
        {
            this.concatenatedMatrices = concatenatedMatrices;
            this.sharedMatrix = sharedMatrix;
            this.DeepOutput = new DeepMatrix(concatenatedMatrices.Depth, 1, concatenatedMatrices.Rows);

            for (int index = 0; index < concatenatedMatrices.Depth; index++)
            {
                Matrix transposed = concatenatedMatrices[index].Transpose();
                this.DeepOutput[index] = sharedMatrix * transposed;
            }

            return this.DeepOutput;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(DeepMatrix dOutput)
        {
            Matrix dSharedMatrix = new Matrix(1, this.sharedMatrix.Cols);
            DeepMatrix dConcatenated = new DeepMatrix(this.concatenatedMatrices.Depth, this.concatenatedMatrices.Rows, this.concatenatedMatrices.Cols);

            for (int index = 0; index < this.concatenatedMatrices.Depth; index++)
            {
                Matrix transposed = this.concatenatedMatrices[index].Transpose();
                Matrix dAttentionCoefficient = dOutput[index];

                // Accumulate gradient for sharedMatrix
                dSharedMatrix += dAttentionCoefficient * transposed.Transpose();

                // Compute gradient for concatenated matrices
                Matrix dTransposed = this.sharedMatrix.Transpose() * dAttentionCoefficient;
                dConcatenated[index] = dTransposed.Transpose();
            }

            return new BackwardResultBuilder()
                .AddDeepInputGradient(dConcatenated)
                .AddWeightGradient(dSharedMatrix)
                .Build();
        }
    }
}
