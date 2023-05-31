//------------------------------------------------------------------------------
// <copyright file="ModelElementType.cs" author="ameritusweb" date="5/27/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// The model element type.
    /// </summary>
    public enum ModelElementType
    {
        /// <summary>
        /// A weight or bias.
        /// </summary>
        Weight,

        /// <summary>
        /// A gradient.
        /// </summary>
        Gradient,

        /// <summary>
        /// The first moment for Adam optimization.
        /// </summary>
        FirstMoment,

        /// <summary>
        /// The second moment for Adam optimization.
        /// </summary>
        SecondMoment,
    }
}
