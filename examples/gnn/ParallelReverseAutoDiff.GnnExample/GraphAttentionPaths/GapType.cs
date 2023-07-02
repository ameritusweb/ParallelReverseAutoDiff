//------------------------------------------------------------------------------
// <copyright file="GapType.cs" author="ameritusweb" date="7/1/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths
{
    /// <summary>
    /// The type of piece on the square.
    /// </summary>
    public enum GapType
    {
        /// <summary>
        /// An empty square.
        /// </summary>
        Empty,

        /// <summary>
        /// The queen.
        /// </summary>
        Queen,

        /// <summary>
        /// The king.
        /// </summary>
        King,

        /// <summary>
        /// The knight.
        /// </summary>
        Knight,

        /// <summary>
        /// The bishop.
        /// </summary>
        Bishop,

        /// <summary>
        /// The rook.
        /// </summary>
        Rook,

        /// <summary>
        /// The pawn.
        /// </summary>
        Pawn,
    }
}
