// ------------------------------------------------------------------------------
// <copyright file="CubeSplitter.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Amaze
{
    /// <summary>
    /// Provides a cube splitter.
    /// </summary>
    public class CubeSplitter
    {
        private const int MaxDepth = 4;

        /// <summary>
        /// Finds the quadrant indices.
        /// </summary>
        /// <param name="coordinate">The coordinate.</param>
        /// <param name="depth">The depth.</param>
        /// <returns>The quadrant indices.</returns>
        public static int[] FindQuadrantIndices(Point3d coordinate, int depth)
        {
            int[] indices = new int[depth];

            if (depth == 0)
            {
                return indices;
            }

            int cubeLength = 10;
            int cubeWidth = 10;
            int cubeHeight = 10;

            double quadrantLength = cubeLength / Math.Pow(2, MaxDepth - depth + 1);
            double quadrantWidth = cubeWidth / Math.Pow(2, MaxDepth - depth + 1);
            double quadrantHeight = cubeHeight / Math.Pow(2, MaxDepth - depth + 1);

            int quadrantIndexX = (int)Math.Floor(coordinate.X / quadrantLength);
            int quadrantIndexY = (int)Math.Floor(coordinate.Y / quadrantWidth);
            int quadrantIndexZ = (int)Math.Floor(coordinate.Z / quadrantHeight);

            indices[0] = GetQuadrantIndex(quadrantIndexX, quadrantIndexY, quadrantIndexZ);

            if (depth > 1)
            {
                Point3d newCoordinate = new Point3d(
                    coordinate.X % quadrantLength,
                    coordinate.Y % quadrantWidth,
                    coordinate.Z % quadrantHeight);

                int[] childIndices = FindQuadrantIndices(newCoordinate, depth - 1);
                Array.Copy(childIndices, 0, indices, 1, depth - 1);
            }

            return indices;
        }

        private static int GetQuadrantIndex(int x, int y, int z)
        {
            return x + (y * 2) + (z * 4);
        }
    }
}
