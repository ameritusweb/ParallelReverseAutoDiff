//------------------------------------------------------------------------------
// <copyright file="IModelLayer.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// Am interface to a model layer.
    /// </summary>
    public interface IModelLayer
    {
        /// <summary>
        /// Gets the model element from the model layer.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <param name="elementType">The element type.</param>
        /// <returns>The model element.</returns>
        object this[string identifier, ModelElementType elementType] { get; }
    }
}
