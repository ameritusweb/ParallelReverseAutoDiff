//------------------------------------------------------------------------------
// <copyright file="EvaluationSquareMessage.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;

    /// <summary>
    /// The evaluation square message.
    /// </summary>
    public class EvaluationSquareMessage
    {
        /// <summary>
        /// Gets or sets the source piece.
        /// </summary>
        public Piece SourcePiece { get; set; }

        /// <summary>
        /// Gets or sets the source position.
        /// </summary>
        public Position SourcePosition { get; set; }

        /// <summary>
        /// Gets or sets the type of the message.
        /// </summary>
        public EvaluationSquareMessageType Type { get; set; }

        /// <summary>
        /// Gets the from direction of the message.
        /// </summary>
        public EvaluationSquareMessageType FromDirection
        {
            get
            {
                return this.Type switch
                {
                    EvaluationSquareMessageType.Top => EvaluationSquareMessageType.Bottom,
                    EvaluationSquareMessageType.Left => EvaluationSquareMessageType.Right,
                    EvaluationSquareMessageType.Bottom => EvaluationSquareMessageType.Top,
                    EvaluationSquareMessageType.Right => EvaluationSquareMessageType.Left,
                    EvaluationSquareMessageType.TopLeft => EvaluationSquareMessageType.BottomRight,
                    EvaluationSquareMessageType.TopRight => EvaluationSquareMessageType.BottomLeft,
                    EvaluationSquareMessageType.BottomLeft => EvaluationSquareMessageType.TopRight,
                    EvaluationSquareMessageType.BottomRight => EvaluationSquareMessageType.TopLeft,
                    _ => throw new NotImplementedException(),
                };
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether to pass along the message.
        /// </summary>
        public bool PassAlong { get; set; }
    }
}
