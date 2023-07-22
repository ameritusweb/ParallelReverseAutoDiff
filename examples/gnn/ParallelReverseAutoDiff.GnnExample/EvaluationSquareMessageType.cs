//------------------------------------------------------------------------------
// <copyright file="EvaluationSquareMessageType.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    /// <summary>
    /// The evaluation square message type.
    /// </summary>
    public enum EvaluationSquareMessageType
    {
        /// <summary>
        /// Moving the piece to the top.
        /// </summary>
        Top,

        /// <summary>
        /// Moving the piece to the left.
        /// </summary>
        Left,

        /// <summary>
        /// Moving the piece to the bottom.
        /// </summary>
        Bottom,

        /// <summary>
        /// Moving the piece to the right.
        /// </summary>
        Right,

        /// <summary>
        /// Moving the piece to the top left.
        /// </summary>
        TopLeft,

        /// <summary>
        /// Moving the piece to the top right.
        /// </summary>
        TopRight,

        /// <summary>
        /// Moving the piece to the bottom left.
        /// </summary>
        BottomLeft,

        /// <summary>
        /// Moving the piece to the bottom right.
        /// </summary>
        BottomRight,
    }
}
