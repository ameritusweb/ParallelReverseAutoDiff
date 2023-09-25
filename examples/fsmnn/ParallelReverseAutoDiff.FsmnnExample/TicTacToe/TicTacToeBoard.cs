//------------------------------------------------------------------------------
// <copyright file="TicTacToeBoard.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.TicTacToe
{
    /// <summary>
    /// Creates a tic tac toe board.
    /// </summary>
    public class TicTacToeBoard
    {
        private char[,] board = new char[3, 3];
        private char currentPlayer;

        /// <summary>
        /// Initializes a new instance of the <see cref="TicTacToeBoard"/> class.
        /// </summary>
        public TicTacToeBoard()
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    this.board[i, j] = ' '; // Initializing the board with empty spaces
                }
            }

            this.currentPlayer = 'X'; // Assuming 'X' starts first
        }

        /// <summary>
        /// Gets the board.
        /// </summary>
        public char[,] Board
        {
            get
            {
                return this.board;
            }
        }

        /// <summary>
        /// Gets the current player.
        /// </summary>
        public char CurrentPlayer
        {
            get
            {
                return this.currentPlayer;
            }
        }

        /// <summary>
        /// Gets the possible moves.
        /// </summary>
        /// <returns>The list of possible moves.</returns>
        public List<TicTacToeBoard> GetPossibleMoves()
        {
            List<TicTacToeBoard> possibleMoves = new List<TicTacToeBoard>();

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (this.board[i, j] == ' ')
                    {
                        TicTacToeBoard newBoard = this.Clone();
                        newBoard.board[i, j] = this.currentPlayer;
                        newBoard.currentPlayer = (this.currentPlayer == 'X') ? 'O' : 'X'; // Switching the player for the next move
                        possibleMoves.Add(newBoard);
                    }
                }
            }

            return possibleMoves;
        }

        /// <summary>
        /// Clones the tic tac tow board.
        /// </summary>
        /// <returns>The tic tac toe board.</returns>
        public TicTacToeBoard Clone()
        {
            TicTacToeBoard newBoard = new TicTacToeBoard();
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    newBoard.board[i, j] = this.board[i, j];
                }
            }

            newBoard.currentPlayer = this.currentPlayer;
            return newBoard;
        }

        /// <summary>
        /// Determines whether the board is terminal.
        /// </summary>
        /// <returns>True or false.</returns>
        public bool IsTerminal()
        {
            // Check rows, columns, diagonals for a win
            for (int i = 0; i < 3; i++)
            {
                if (this.board[i, 0] == this.board[i, 1] && this.board[i, 1] == this.board[i, 2] && this.board[i, 0] != ' ')
                {
                    return true;
                }

                if (this.board[0, i] == this.board[1, i] && this.board[1, i] == this.board[2, i] && this.board[0, i] != ' ')
                {
                    return true;
                }
            }

            if (this.board[0, 0] == this.board[1, 1] && this.board[1, 1] == this.board[2, 2] && this.board[0, 0] != ' ')
            {
                return true;
            }

            if (this.board[0, 2] == this.board[1, 1] && this.board[1, 1] == this.board[2, 0] && this.board[0, 2] != ' ')
            {
                return true;
            }

            // Check for a full board (tie)
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (this.board[i, j] == ' ')
                    {
                        return false; // if there's an empty spot, not a terminal state
                    }
                }
            }

            // Board is full, so it's a tie
            return true;
        }
    }
}
