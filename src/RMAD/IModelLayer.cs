//------------------------------------------------------------------------------
// <copyright file="IModelLayer.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

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
        Matrix? WeightMatrix(string identifier);

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public DeepMatrix? WeightDeepMatrix(string identifier);

        /// <summary>
        /// Retrieve the weight by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The weight.</returns>
        public DeepMatrix[]? WeightDeepMatrixArray(string identifier);

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public Matrix? GradientMatrix(string identifier);

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public DeepMatrix? GradientDeepMatrix(string identifier);

        /// <summary>
        /// Retrieve the gradient by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The gradient.</returns>
        public DeepMatrix[]? GradientDeepMatrixArray(string identifier);

        /// <summary>
        /// Retrieve the dimensions by identifier.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The dimensions.</returns>
        public int[]? Dimensions(string identifier);
    }
}
