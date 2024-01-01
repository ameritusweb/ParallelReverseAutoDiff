//------------------------------------------------------------------------------
// <copyright file="IModelLayer.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;
    using System.IO;

    /// <summary>
    /// Am interface to a model layer.
    /// </summary>
    public interface IModelLayer
    {
        /// <summary>
        /// Gets identifiers.
        /// </summary>
        public List<string> Identifiers { get; }

        /// <summary>
        /// Gets the model element from the model layer.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <param name="elementType">The element type.</param>
        /// <returns>The model element.</returns>
        object this[string identifier, ModelElementType elementType] { get; }

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public Matrix WeightMatrix(string identifier);

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public DeepMatrix WeightDeepMatrix(string identifier);

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public DeepMatrix[] WeightDeepMatrixArray(string identifier);

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public Matrix GradientMatrix(string identifier);

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public DeepMatrix GradientDeepMatrix(string identifier);

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public DeepMatrix[] GradientDeepMatrixArray(string identifier);

        /// <summary>
        /// Retrieve the dimensions by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The dimensions.</returns>
        public int[] Dimensions(string identifier);

        /// <summary>
        /// Randomizes the weights.
        /// </summary>
        /// <returns>The randomized weights.</returns>
        public IModelLayer RandomizeWeights();

        /// <summary>
        /// Clones the model layer.
        /// </summary>
        /// <returns>The model layer.</returns>
        public IModelLayer Clone();

        /// <summary>
        /// Applies the gradients.
        /// </summary>
        /// <param name="gradients">The gradients to apply.</param>
        public void ApplyGradients(List<object> gradients);

        /// <summary>
        /// Applies the weights.
        /// </summary>
        /// <param name="weights">The weights to apply.</param>
        public void ApplyWeights(List<object> weights);

        /// <summary>
        /// Takes the average of two model layers.
        /// </summary>
        /// <param name="layer">The other model layer.</param>
        public void Average(IModelLayer layer);

        /// <summary>
        /// Averages the gradients of two model layers.
        /// </summary>
        /// <param name="layer">The other model layer.</param>
        public void AverageGradients(IModelLayer layer);

        /// <summary>
        /// Save the weights to a file.
        /// </summary>
        /// <param name="file">The file info.</param>
        public void SaveWeights(FileInfo file);

        /// <summary>
        /// Load the weights from a file.
        /// </summary>
        /// <param name="file">The file info.</param>
        public void LoadWeights(FileInfo file);

        /// <summary>
        /// Save the weights and moments to a binary file.
        /// If binary serialization is unsuccessful, it will fall back to JSON serialization.
        /// </summary>
        /// <param name="file">The file info.</param>
        public void SaveWeightsAndMomentsBinary(FileInfo file);

        /// <summary>
        /// Load the weights and moments from a binary file.
        /// </summary>
        /// <param name="file">The file info.</param>
        public void LoadWeightsAndMomentsBinary(FileInfo file);

        /// <summary>
        /// Save the weights and moments to a file.
        /// </summary>
        /// <param name="file">The file info.</param>
        public void SaveWeightsAndMoments(FileInfo file);

        /// <summary>
        /// Load the weights and moments from a file.
        /// </summary>
        /// <param name="file">The file info.</param>
        public void LoadWeightsAndMoments(FileInfo file);
    }
}
