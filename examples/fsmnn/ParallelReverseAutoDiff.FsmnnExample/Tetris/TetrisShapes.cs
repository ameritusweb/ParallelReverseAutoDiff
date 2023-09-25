//------------------------------------------------------------------------------
// <copyright file="TetrisShapes.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Tetris
{
    /// <summary>
    /// Tetris shapes.
    /// </summary>
    public class TetrisShapes
    {
        private static readonly bool[,,,] Shapes = new bool[7, 4, 4, 4]
        {
            // I (Line) Piece
            {
                {
                    { false, false, false, false },
                    { true,  true,  true,  true },
                    { false, false, false, false },
                    { false, false, false, false },
                },
                {
                    { false, false, true, false },
                    { false, false, true, false },
                    { false, false, true, false },
                    { false, false, true, false },
                },
                {
                    { false, false, false, false },
                    { true,  true,  true,  true },
                    { false, false, false, false },
                    { false, false, false, false },
                },
                {
                    { false, false, true, false },
                    { false, false, true, false },
                    { false, false, true, false },
                    { false, false, true, false },
                },
            },

            // O (Square) Piece
            {
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
            },

            // T Piece
            {
                // Default (0°)
                {
                    { false, false, false, false },
                    { false, true,  true,  true },
                    { false, false, true,  false },
                    { false, false, false, false },
                },

                // 90°
                {
                    { false, false, true,  false },
                    { false, false, true,  true },
                    { false, false, true,  false },
                    { false, false, false, false },
                },

                // 180°
                {
                    { false, false, false, false },
                    { false, false, true,  false },
                    { false, true,  true,  true },
                    { false, false, false, false },
                },

                // 270°
                {
                    { false, false, true,  false },
                    { false, true,  true,  false },
                    { false, false, true,  false },
                    { false, false, false, false },
                },
            },
            {
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { true,  true,  false, false },
                    { false, false, false, false },
                },
                {
                    { false, true,  false, false },
                    { false, true,  true,  false },
                    { false, false, true,  false },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { true,  true,  false, false },
                    { false, false, false, false },
                },
                {
                    { false, true,  false, false },
                    { false, true,  true,  false },
                    { false, false, true,  false },
                    { false, false, false, false },
                },
            },

            // Z Piece
            {
                {
                    { false, false, false, false },
                    { true,  true,  false, false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
                {
                    { false, false, true,  false },
                    { false, true,  true,  false },
                    { false, true,  false, false },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { true,  true,  false, false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
                {
                    { false, false, true,  false },
                    { false, true,  true,  false },
                    { false, true,  false, false },
                    { false, false, false, false },
                },
            },

            // J Piece
            {
                {
                    { false, false, false, false },
                    { true,  true,  true,  false },
                    { false, false, true,  false },
                    { false, false, false, false },
                },
                {
                    { false, true,  false, false },
                    { false, true,  false, false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { false, true,  false, false },
                    { false, true,  true,  true },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { false, true,  false, false },
                    { false, true,  false, false },
                },
            },

            // L Piece
            {
                {
                    { false, false, false, false },
                    { false, true,  true,  true },
                    { false, true,  false, false },
                    { false, false, false, false },
                },
                {
                    { false, false, false, false },
                    { false, true,  true,  false },
                    { false, false, true,  false },
                    { false, false, true,  false },
                },
                {
                    { false, false, false, false },
                    { false, false, true,  false },
                    { true,  true,  true,  false },
                    { false, false, false, false },
                },
                {
                    { false, true,  false, false },
                    { false, true,  false, false },
                    { false, true,  true,  false },
                    { false, false, false, false },
                },
            },
        };

        /// <summary>
        /// Gets the tetris shapes.
        /// </summary>
        /// <param name="p1">Param 1.</param>
        /// <param name="p2">Param 2.</param>
        /// <param name="p3">Param 3.</param>
        /// <param name="p4">Param 4.</param>
        /// <returns>True or false.</returns>
        public bool this[int p1, int p2, int p3, int p4]
        {
            get
            {
                return Shapes[p1, p2, p3, p4];
            }
        }
    }
}
