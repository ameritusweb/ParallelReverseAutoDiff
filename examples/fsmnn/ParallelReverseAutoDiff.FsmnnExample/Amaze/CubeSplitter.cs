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
        private int maxDepth;

        /// <summary>
        /// Initializes a new instance of the <see cref="CubeSplitter"/> class.
        /// </summary>
        /// <param name="maxDepth">The max depth.</param>
        public CubeSplitter(int maxDepth)
        {
            this.maxDepth = maxDepth;
        }

        /// <summary>
        /// Finds the quadrant indices.
        /// </summary>
        /// <param name="coordinate">The coordinate.</param>
        /// <returns>The quadrant indices.</returns>
        public int[] FindQuadrantIndices(Point3d coordinate)
        {
            return this.FindQuadrantIndices(coordinate, this.maxDepth);
        }

        /// <summary>
        /// Finds the quadrant indices.
        /// </summary>
        /// <param name="coordinate">The coordinate.</param>
        /// <param name="depth">The depth.</param>
        /// <returns>The quadrant indices.</returns>
        public int[] FindQuadrantIndices(Point3d coordinate, int depth)
        {
            int[] indices = new int[depth];

            if (depth == 0)
            {
                return indices;
            }

            int cubeLength = 10;
            int cubeWidth = 10;
            int cubeHeight = 10;

            double quadrantLength = cubeLength / Math.Pow(2, this.maxDepth - depth + 1);
            double quadrantWidth = cubeWidth / Math.Pow(2, this.maxDepth - depth + 1);
            double quadrantHeight = cubeHeight / Math.Pow(2, this.maxDepth - depth + 1);

            int quadrantIndexX = (int)Math.Floor(coordinate.X / quadrantLength);
            int quadrantIndexY = (int)Math.Floor(coordinate.Y / quadrantWidth);
            int quadrantIndexZ = (int)Math.Floor(coordinate.Z / quadrantHeight);

            indices[0] = this.GetQuadrantIndex(quadrantIndexX, quadrantIndexY, quadrantIndexZ);

            if (depth > 1)
            {
                Point3d newCoordinate = new Point3d(
                    coordinate.X % quadrantLength,
                    coordinate.Y % quadrantWidth,
                    coordinate.Z % quadrantHeight);

                int[] childIndices = this.FindQuadrantIndices(newCoordinate, depth - 1);
                Array.Copy(childIndices, 0, indices, 1, depth - 1);
            }

            return indices;
        }

        private int GetQuadrantIndex(int x, int y, int z)
        {
            return x + (y * 2) + (z * 4);
        }
    }
}
