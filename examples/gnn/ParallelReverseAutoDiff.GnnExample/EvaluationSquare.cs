//------------------------------------------------------------------------------
// <copyright file="EvaluationSquare.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;

    /// <summary>
    /// The evaluation square.
    /// </summary>
    public class EvaluationSquare
    {
        /// <summary>
        /// Gets or sets the left neighbor.
        /// </summary>
        public EvaluationSquare? Left { get; set; }

        /// <summary>
        /// Gets or sets the top left neighbor.
        /// </summary>
        public EvaluationSquare? TopLeft { get; set; }

        /// <summary>
        /// Gets or sets the top neighbor.
        /// </summary>
        public EvaluationSquare? Top { get; set; }

        /// <summary>
        /// Gets or sets the right neighbor.
        /// </summary>
        public EvaluationSquare? Right { get; set; }

        /// <summary>
        /// Gets or sets the top right neighbor.
        /// </summary>
        public EvaluationSquare? TopRight { get; set; }

        /// <summary>
        /// Gets or sets the bottom neighbor.
        /// </summary>
        public EvaluationSquare? Bottom { get; set; }

        /// <summary>
        /// Gets or sets the bottom left neighbor.
        /// </summary>
        public EvaluationSquare? BottomLeft { get; set; }

        /// <summary>
        /// Gets or sets the bottom right neighbor.
        /// </summary>
        public EvaluationSquare? BottomRight { get; set; }

        /// <summary>
        /// Gets or sets the position.
        /// </summary>
        public Position Position { get; set; }

        /// <summary>
        /// Gets or sets the piece.
        /// </summary>
        public Piece? Piece { get; set; }

        /// <summary>
        /// Gets or sets the received messages.
        /// </summary>
        public List<EvaluationSquareMessage> ReceivedMessages { get; set; } = new List<EvaluationSquareMessage>();

        /// <summary>
        /// Gets the white messages.
        /// </summary>
        public List<EvaluationSquareMessage> WhiteMessages
        {
            get
            {
                return this.ReceivedMessages.Where(x => x.SourcePiece.Color == PieceColor.White).ToList();
            }
        }

        /// <summary>
        /// Gets the black messages.
        /// </summary>
        public List<EvaluationSquareMessage> BlackMessages
        {
            get
            {
                return this.ReceivedMessages.Where(x => x.SourcePiece.Color == PieceColor.Black).ToList();
            }
        }

        /// <summary>
        /// Gets the white enum counts.
        /// </summary>
        public Dictionary<EvaluationSquareMessageType, int> WhiteEnumCounts
        {
            get
            {
                Dictionary<EvaluationSquareMessageType, int> counts = new Dictionary<EvaluationSquareMessageType, int>();
                foreach (EvaluationSquareMessageType type in Enum.GetValues(typeof(EvaluationSquareMessageType)))
                {
                    var count = this.WhiteMessages.Count(m => m.Type == type);
                    if (count > 0)
                    {
                        counts.Add(type, count);
                    }
                }

                return counts;
            }
        }

        /// <summary>
        /// Gets the black enum counts.
        /// </summary>
        public Dictionary<EvaluationSquareMessageType, int> BlackEnumCounts
        {
            get
            {
                Dictionary<EvaluationSquareMessageType, int> counts = new Dictionary<EvaluationSquareMessageType, int>();
                foreach (EvaluationSquareMessageType type in Enum.GetValues(typeof(EvaluationSquareMessageType)))
                {
                    var count = this.BlackMessages.Count(m => m.Type == type);
                    if (count > 0)
                    {
                        counts.Add(type, count);
                    }
                }

                return counts;
            }
        }

        /// <summary>
        /// Gets the max white enum count.
        /// </summary>
        public int MaxWhiteEnumCount
        {
            get
            {
                if (this.WhiteEnumCounts.Count == 0)
                {
                    return 0;
                }

                return this.WhiteEnumCounts.Values.Max();
            }
        }

        /// <summary>
        /// Gets the max black enum count.
        /// </summary>
        public int MaxBlackEnumCount
        {
            get
            {
                if (this.BlackEnumCounts.Count == 0)
                {
                    return 0;
                }

                return this.BlackEnumCounts.Values.Max();
            }
        }

        /// <summary>
        /// Set the neighbors.
        /// </summary>
        /// <param name="top">The top neighbor.</param>
        /// <param name="left">The left neighbor.</param>
        /// <param name="bottom">The bottom neighbor.</param>
        /// <param name="right">The right neighbor.</param>
        /// <param name="topLeft">The top left neighbor.</param>
        /// <param name="topRight">The top right neighbor.</param>
        /// <param name="bottomLeft">The bottom left neighbor.</param>
        /// <param name="bottomRight">The bottom right neigbor.</param>
        public void SetNeighbors(
            EvaluationSquare? top,
            EvaluationSquare? left,
            EvaluationSquare? bottom,
            EvaluationSquare? right,
            EvaluationSquare? topLeft,
            EvaluationSquare? topRight,
            EvaluationSquare? bottomLeft,
            EvaluationSquare? bottomRight)
        {
            this.Top = top;
            this.Left = left;
            this.Bottom = bottom;
            this.Right = right;
            this.TopLeft = topLeft;
            this.TopRight = topRight;
            this.BottomLeft = bottomLeft;
            this.BottomRight = bottomRight;
        }

        /// <summary>
        /// Receive the message.
        /// </summary>
        /// <param name="message">The message to receive.</param>
        public void ReceiveMessage(EvaluationSquareMessage message)
        {
            this.ReceivedMessages.Add(message);
            if (message.PassAlong && !(this.Piece != null
                && (this.Piece.Color != message.SourcePiece.Color || this.Piece.Type == PieceType.Pawn || this.Piece.Type == PieceType.King || this.Piece.Type == PieceType.Knight)))
            {
                if (this.Piece != null && (message.Type == EvaluationSquareMessageType.Left || message.Type == EvaluationSquareMessageType.Right))
                {
                    return;
                }

                if (this.Piece != null && this.Piece.MaterialValue > message.SourcePiece.MaterialValue)
                {
                    return;
                }

                if (this.Piece != null && this.Piece.Type == PieceType.Bishop)
                {
                    if (message.Type == EvaluationSquareMessageType.Top
                        ||
                        message.Type == EvaluationSquareMessageType.Bottom
                        ||
                        message.Type == EvaluationSquareMessageType.Left
                        ||
                        message.Type == EvaluationSquareMessageType.Right)
                    {
                        return;
                    }
                }

                if (this.Piece != null && this.Piece.Type == PieceType.Rook)
                {
                    if (message.Type == EvaluationSquareMessageType.TopLeft
                        ||
                        message.Type == EvaluationSquareMessageType.TopRight
                        ||
                        message.Type == EvaluationSquareMessageType.BottomLeft
                        ||
                        message.Type == EvaluationSquareMessageType.BottomRight)
                    {
                        return;
                    }
                }

                this.PassMessage(message.Type, message.PassAlong, message);
            }
        }

        /// <summary>
        /// Pass a message to a neighbor.
        /// </summary>
        /// <param name="messageType">The message type.</param>
        /// <param name="passAlong">A value indicating whether to pass the message along.</param>
        /// <param name="message">The message to pass.</param>
        public void PassMessage(EvaluationSquareMessageType messageType, bool passAlong, EvaluationSquareMessage message)
        {
            message.Type = messageType;
            message.PassAlong = passAlong;
            switch (messageType)
            {
                case EvaluationSquareMessageType.Top:
                    this.Top?.ReceiveMessage(message);
                    break;
                case EvaluationSquareMessageType.Left:
                    this.Left?.ReceiveMessage(message);
                    break;
                case EvaluationSquareMessageType.Right:
                    this.Right?.ReceiveMessage(message);
                    break;
                case EvaluationSquareMessageType.Bottom:
                    this.Bottom?.ReceiveMessage(message);
                    break;
                case EvaluationSquareMessageType.TopLeft:
                    this.TopLeft?.ReceiveMessage(message);
                    break;
                case EvaluationSquareMessageType.TopRight:
                    this.TopRight?.ReceiveMessage(message);
                    break;
                case EvaluationSquareMessageType.BottomLeft:
                    this.BottomLeft?.ReceiveMessage(message);
                    break;
                case EvaluationSquareMessageType.BottomRight:
                    this.BottomRight?.ReceiveMessage(message);
                    break;
                default:
                    break;
            }
        }

        /// <summary>
        /// Pass the messages along.
        /// </summary>
        public void PassMessages()
        {
            if (this.Piece != null)
            {
                List<EvaluationSquareMessageType> messageTypes = new List<EvaluationSquareMessageType>();

                switch (this.Piece.Type.Name)
                {
                    case nameof(PieceType.Rook):
                        messageTypes = new List<EvaluationSquareMessageType>
                        {
                            EvaluationSquareMessageType.Top,
                            EvaluationSquareMessageType.Bottom,
                            EvaluationSquareMessageType.Left,
                            EvaluationSquareMessageType.Right,
                        };
                        break;
                    case nameof(PieceType.Bishop):
                        messageTypes = new List<EvaluationSquareMessageType>
                        {
                            EvaluationSquareMessageType.TopLeft,
                            EvaluationSquareMessageType.TopRight,
                            EvaluationSquareMessageType.BottomLeft,
                            EvaluationSquareMessageType.BottomRight,
                        };
                        break;
                    case nameof(PieceType.Queen):
                        messageTypes = Enum.GetValues(typeof(EvaluationSquareMessageType)).Cast<EvaluationSquareMessageType>().ToList();
                        break;
                    default:
                        break;
                }

                foreach (var messageType in messageTypes)
                {
                    var message = new EvaluationSquareMessage() { SourcePiece = this.Piece, SourcePosition = this.Position };
                    this.PassMessage(messageType, true, message);
                }
            }
        }
    }
}
