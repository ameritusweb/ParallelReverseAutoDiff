// *****************************************************
// *                                                   *
// * O Lord, Thank you for your goodness in our lives. *
// *     Please bless this code to our compilers.      *
// *                     Amen.                         *
// *                                                   *
// *****************************************************
//                                    Made by Geras1mleo

using System.Collections.Generic;

namespace Chess
{
    /// <summary>
    /// Position on Chess table counting from 0
    /// </summary>
    public struct Position
    {
        public static Position Empty
        {
            get
            {
                return new Position(-1, -1);
            }
        }

        /// <summary>
        /// Whether X is between 0 and 7<br/>
        /// And Y is between 0 and 7
        /// </summary>
        public bool HasValue => HasValueX & HasValueY;

        /// <summary>
        /// Whether X is between 0 and 7
        /// </summary>
        public bool HasValueX => X >= 0 && X <= 7;
        /// <summary>
        /// Whether Y is between 0 and 7
        /// </summary>
        public bool HasValueY => Y >= 0 && Y <= 7;

        /// <summary>
        /// Horizontal position (File) on chess board
        /// </summary>
        public short X { get; internal set; }
        /// <summary>
        /// Vertical position (Rank) on chess board
        /// </summary>
        public short Y { get; internal set; }

        /// <summary>
        /// Get the rank on chess board.
        /// </summary>
        public short RankValue => Y;

        /// <summary>
        /// Get the file on chess board.
        /// </summary>
        public short FileValue => X;

        /// <summary>
        /// Initializes a new Position ex.:<br/>
        /// "a1" - notation => {X = 0, Y = 0}<br/>
        /// "h8" - notation => {X = 7, Y = 7}<br/>
        /// </summary>
        /// <param name="position">Position as string</param>
        public Position(string position)
        {
            position = position.ToLower();

            if (!Regexes.RegexPosition.IsMatch(position))
                throw new ChessArgumentException(null, "Table position should match pattern: " + Regexes.PositionPattern);

            X = FromFile(position[0]);
            Y = FromRank(position[1]);
        }

        /// <summary>
        /// Initializes a new Position in chess board<br/>
        /// Counting from 0 
        /// </summary>
        public Position(short x, short y)
        {
            X = x;
            Y = y;
        }

        public Position(int rank, int file)
        {
            X = (short)file;
            Y = (short)rank;
        }

        /// <summary>
        /// Short horizontal position from file char<br/>
        /// 'a' => 0<br/>
        /// 'b' => 1<br/>
        /// 'c' => 2<br/>
        /// 'd' => 3<br/>
        /// 'e' => 4<br/>
        /// 'f' => 5<br/>
        /// 'g' => 6<br/>
        /// 'h' => 7<br/>
        /// </summary>
        public static short FromFile(char file)
        {
            return char.ToLower(file) switch
            {
                'a' => 0,
                'b' => 1,
                'c' => 2,
                'd' => 3,
                'e' => 4,
                'f' => 5,
                'g' => 6,
                'h' => 7,
                _ => throw new ChessArgumentException(null, nameof(file), nameof(Position.FromFile)),
            };
        }

        /// <summary>
        /// Short vertical position from rank char<br/>
        /// '1' => 0<br/>
        /// '2' => 1<br/>
        /// '3' => 2<br/>
        /// '4' => 3<br/>
        /// '5' => 4<br/>
        /// '6' => 5<br/>
        /// '7' => 6<br/>
        /// '8' => 7<br/>
        /// </summary>
        public static short FromRank(char rank)
        {   // This code is faster than conversion to short with short.TryParse(...)
            return rank switch
            {
                '1' => 0,
                '2' => 1,
                '3' => 2,
                '4' => 3,
                '5' => 4,
                '6' => 5,
                '7' => 6,
                '8' => 7,
                _ => throw new ChessArgumentException(null, nameof(rank), nameof(Position.FromRank)),
            };
        }

        public Position MoveBy(int rank, int file)
        {
            return new Position { X = ((short)(this.X + file)), Y = ((short)(this.Y + rank)) };
        }

        public IEnumerable<Position> GetAdjacentSquares()
        {
            var positions = new List<Position>();
            if (X > 0)
            {
                if (Y > 0) positions.Add(new Position((short)(X - 1), (short)(Y - 1)));
                positions.Add(new Position((short)(X - 1), Y));
                if (Y < 7) positions.Add(new Position((short)(X - 1), (short)(Y + 1)));
            }
            if (Y > 0) positions.Add(new Position(X, (short)(Y - 1)));
            if (Y < 7) positions.Add(new Position(X, (short)(Y + 1)));
            if (X < 7)
            {
                if (Y > 0) positions.Add(new Position((short)(X + 1), (short)(Y - 1)));
                positions.Add(new Position((short)(X + 1), Y));
                if (Y < 7) positions.Add(new Position((short)(X + 1), (short)(Y + 1)));
            }
            return positions;
        }

        /// <summary>
        /// Char from X<br/>
        /// 0 => 'a'<br/>
        /// 1 => 'b'<br/>
        /// 2 => 'c'<br/>
        /// 3 => 'd'<br/>
        /// 4 => 'e'<br/>
        /// 5 => 'f'<br/>
        /// 6 => 'g'<br/>
        /// 7 => 'h'<br/>
        /// </summary>
        public char File()
        {
            return X switch
            {
                0 => 'a',
                1 => 'b',
                2 => 'c',
                3 => 'd',
                4 => 'e',
                5 => 'f',
                6 => 'g',
                7 => 'h',
                _ => throw new ChessArgumentException(null, nameof(X), nameof(Position.File))
            };
        }

        /// <summary>
        /// Char from Y<br/>
        /// 0 => '1'<br/>
        /// 1 => '2'<br/>
        /// 2 => '3'<br/>
        /// 3 => '4'<br/>
        /// 4 => '5'<br/>
        /// 5 => '6'<br/>
        /// 6 => '7'<br/>
        /// 7 => '8'<br/>
        /// </summary>
        public char Rank()
        {   // This code is faster than conversion to char
            return Y switch
            {
                0 => '1',
                1 => '2',
                2 => '3',
                3 => '4',
                4 => '5',
                5 => '6',
                6 => '7',
                7 => '8',
                _ => throw new ChessArgumentException(null, nameof(Y), nameof(Position.Rank))
            };
        }

        /// <summary>
        /// Position as string position on board with rank:<br/>
        /// {X = 0, Y = 0} => "a1"<br/>
        /// {X = 7, Y = 7} => "h8"<br/>
        /// </summary>
        public override string ToString() => File().ToString() + Rank();

        /// <summary>
        /// Equals
        /// </summary>
        public override bool Equals(object obj) => base.Equals(obj);
        /// <summary>
        /// HashCode
        /// </summary>
        public override int GetHashCode() => base.GetHashCode();

        /// <summary>
        /// Equalizing 2 Positions
        /// </summary>
        public static bool operator ==(Position a, Position b) => (a.X == b.X && a.Y == b.Y);
        /// <summary>
        /// Equalizing 2 Positions
        /// </summary>
        public static bool operator !=(Position a, Position b) => !(a.X == b.X && a.Y == b.Y);
    }
}