//------------------------------------------------------------------------------
// <copyright file="CosineScalingOperation.cs" author="ameritusweb" date="9/4/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    /// <summary>
    /// Custom cosine scaling operation.
    /// </summary>
    public class CosineScalingOperation : Operation
    {
        private Matrix learnedVectors;
        private Matrix inputMatrix;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new CosineScalingOperation();
        }

        /// <inheritdoc />
        public override void Store(Guid id)
        {
            // Store the intermediate matrices
            this.IntermediateMatrices.AddOrUpdate(id, this.inputMatrix, (x, y) => this.inputMatrix);
        }

        /// <inheritdoc />
        public override void Restore(Guid id)
        {
            // Restore the intermediate matrices
            this.inputMatrix = this.IntermediateMatrices[id];
        }

        /// <summary>
        /// Performs the forward operation.
        /// </summary>
        /// <param name="inputMatrix">MxN input matrix.</param>
        /// <param name="learnedVectors">MxN learned vectors.</param>
        /// <returns>MxN scaled matrix.</returns>
        public Matrix Forward(Matrix inputMatrix, Matrix learnedVectors)
        {
            this.inputMatrix = inputMatrix;
            this.learnedVectors = learnedVectors;

            this.Output = new Matrix(inputMatrix.Rows, inputMatrix.Cols);
            for (int i = 0; i < inputMatrix.Cols; i++)
            {
                Matrix slice = inputMatrix.ColumnSlice(i);
                Matrix learnedSlice = learnedVectors.ColumnSlice(i);
                double cosineSim = slice.CosineSimilarity(learnedSlice);
                double scale = 1.0 - cosineSim;

                Matrix scaledSlice = slice * scale;
                this.Output.SetColumnSlice(i, scaledSlice);
            }

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            Matrix dInput = new Matrix(this.inputMatrix.Rows, this.inputMatrix.Cols);
            Matrix dLearnedVectors = new Matrix(this.learnedVectors.Rows, this.learnedVectors.Cols);

            for (int i = 0; i < this.inputMatrix.Cols; i++)
            {
                Matrix slice = this.inputMatrix.ColumnSlice(i);
                Matrix learnedSlice = this.learnedVectors.ColumnSlice(i);
                double cosineSim = slice.CosineSimilarity(learnedSlice);
                double scale = 1.0 - cosineSim;

                Matrix dSlice = dOutput.ColumnSlice(i) * scale;
                Matrix gradCosineInput = slice.GradientWRTCosineSimilarity(learnedSlice, -dSlice.Sum());
                Matrix gradCosineLearned = learnedSlice.GradientWRTCosineSimilarity(slice, -dSlice.Sum());

                dInput.SetColumnSlice(i, dSlice + gradCosineInput);
                dLearnedVectors.SetColumnSlice(i, gradCosineLearned);
            }

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .AddInputGradient(dLearnedVectors)
                .Build();
        }
    }
}
