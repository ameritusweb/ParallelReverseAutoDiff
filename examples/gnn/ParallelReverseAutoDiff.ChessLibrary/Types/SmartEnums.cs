// *****************************************************
// *                                                   *
// * O Lord, Thank you for your goodness in our lives. *
// *     Please bless this code to our compilers.      *
// *                     Amen.                         *
// *                                                   *
// *****************************************************
//                                    Made by Geras1mleo

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

using Ardalis.SmartEnum;

namespace Chess
{
    /// <summary>
    /// Smart enum for Piece color in chess game
    /// </summary>
    public class PieceColor : SmartEnum<PieceColor>
    {
        public static readonly PieceColor White = new("White", 1, 'w');
        public static readonly PieceColor Black = new("Black", 2, 'b');

        /// <summary>
        /// White => 'w'<br/>
        /// Black => 'b'<br/>
        /// </summary>
        public char AsChar { get; }

        /// <summary>
        /// Opposite color <br/>
        /// White => Black<br/>
        /// Black => White<br/>
        /// </summary>
        public PieceColor OppositeColor()
        {
            return Value switch
            {
                1 => Black,
                2 => White,
                _ => throw new ChessArgumentException(null, nameof(Value), nameof(PieceColor.OppositeColor)),
            };
        }

        private PieceColor(string name, int value, char asChar) : base(name, value)
        {
            AsChar = asChar;
        }

        /// <summary>
        /// PieceColor object from char<br/>
        /// 'w' => White<br/>
        /// 'b' => Black<br/>
        /// </summary>
        public static PieceColor FromChar(char color)
        {
            return char.ToLower(color) switch
            {
                'w' => White,
                'b' => Black,
                _ => throw new ChessArgumentException(null, nameof(color), nameof(PieceColor.FromChar)),
            };
        }
    }

    /// <summary>
    /// Smart enum for Piece type in chess game
    /// </summary>
    public class PieceType : SmartEnum<PieceType>
    {
        public static readonly PieceType Pawn = new("Pawn", 1, 'p');
        public static readonly PieceType Rook = new("Rook", 2, 'r');
        public static readonly PieceType Knight = new("Knight", 3, 'n');
        public static readonly PieceType Bishop = new("Bishop", 4, 'b');
        public static readonly PieceType Queen = new("Queen", 5, 'q');
        public static readonly PieceType King = new("King", 6, 'k');

        /// <summary>
        /// Pawn => 'p'<br/>
        /// Rook => 'r'<br/>
        /// Knight => 'n'<br/>
        /// Bishop => 'b'<br/>
        /// Queen => 'q'<br/>
        /// King => 'k'<br/>
        /// </summary>
        public char AsChar { get; }

        private PieceType(string name, int value, char asChar) : base(name, value)
        {
            AsChar = asChar;
        }

        /// <summary>
        /// PieceType object from char<br/>
        /// 'p' => Pawn<br/>
        /// 'r' => Rook<br/>
        /// 'n' => Knight<br/>
        /// 'b' => Bishop<br/>
        /// 'q' => Queen<br/>
        /// 'k' => King<br/>
        /// </summary>
        public static PieceType FromChar(char type)
        {
            return char.ToLower(type) switch
            {
                'p' => Pawn,
                'r' => Rook,
                'n' => Knight,
                'b' => Bishop,
                'q' => Queen,
                'k' => King,
                _ => throw new ChessArgumentException(null, nameof(type), nameof(PieceType.FromChar)),
            };
        }
    }

    /// <summary>
    /// Smart enum for Move type in chess game
    /// </summary>
    public class MoveType : SmartEnum<MoveType>
    {
        public static readonly MoveType None = new("None", 1, 'n');
        public static readonly MoveType Capture = new("Capture", 2, 'x');
        public static readonly MoveType QueensideCastle = new("QueensideCastle", 3, 'q');
        public static readonly MoveType KingsideCastle = new("KingsideCastle", 4, 'k');
        public static readonly MoveType Promotion = new("Promotion", 5, 'p');
        public static readonly MoveType EnPassant = new("EnPassant", 6, 'e');
        public static readonly MoveType Defense = new("Defense", 7, 'd');

        /// <summary>
        /// None => 'n'<br/>
        /// Capture => 'x'<br/>
        /// QueensideCastle => 'q'<br/>
        /// KingsideCastle => 'k'<br/>
        /// Promotion => 'p'<br/>
        /// EnPassant => 'e'<br/>
        /// Defense => 'd'<br/>
        /// </summary>
        public char AsChar { get; }

        private MoveType(string name, int value, char asChar) : base(name, value)
        {
            AsChar = asChar;
        }

        /// <summary>
        /// PieceType object from char<br/>
        /// 'n' => None<br/>
        /// 'x' => Capture<br/>
        /// 'q' => QueensideCastle<br/>
        /// 'k' => KingsideCastle<br/>
        /// 'p' => Promotion<br/>
        /// 'e' => EnPassant<br/>
        /// 'd' => Defense<br/>
        /// </summary>
        public static MoveType FromChar(char type)
        {
            return char.ToLower(type) switch
            {
                'n' => None,
                'x' => Capture,
                'q' => QueensideCastle,
                'k' => KingsideCastle,
                'p' => Promotion,
                'e' => EnPassant,
                'd' => Defense,
                _ => throw new ChessArgumentException(null, nameof(type), nameof(MoveType.FromChar)),
            };
        }
    }
}

#pragma warning restore CS1591 // Missing XML comment for publicly visible type or member