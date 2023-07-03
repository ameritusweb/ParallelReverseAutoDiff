//------------------------------------------------------------------------------
// <copyright file="ChessBoardLoader.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using Chess;

    /// <summary>
    /// A chess board loader.
    /// </summary>
    public class ChessBoardLoader
    {
        private List<string> files;

        /// <summary>
        /// Initializes a new instance of the <see cref="ChessBoardLoader"/> class.
        /// </summary>
        public ChessBoardLoader()
        {
            this.files = Directory.EnumerateFiles(Directory.GetParent(Directory.GetCurrentDirectory())?.Parent?.Parent?.FullName + "\\PGNLibrary", "*.pgn", SearchOption.AllDirectories).ToList();
        }

        /// <summary>
        /// Gets the total.
        /// </summary>
        /// <returns>The total.</returns>
        public int GetTotal()
        {
            return this.files.Count;
        }

        /// <summary>
        /// Get the file name.
        /// </summary>
        /// <param name="skip">How many files to skip.</param>
        /// <returns>The file name.</returns>
        public string GetFileName(int skip)
        {
            string file = this.files.Skip(skip).First();
            FileInfo fileInfo = new FileInfo(file);
            return fileInfo.Name;
        }

        /// <summary>
        /// Load the chess board.
        /// </summary>
        /// <param name="skip">How many files to skip.</param>
        /// <returns>The chess board list of moves.</returns>
        public List<Move> LoadMoves(int skip)
        {
            string file = this.files.Skip(skip).First();
            string text = File.ReadAllText(file);
            text = text.Replace("\r\n", " ");
            ChessBoard.TryLoadFromPgn(text, out ChessBoard? chessBoard);
            var board = chessBoard ?? throw new InvalidOperationException("Chess Board could not be loaded.");
            return board.ExecutedMoves.ToList();
        }
    }
}
