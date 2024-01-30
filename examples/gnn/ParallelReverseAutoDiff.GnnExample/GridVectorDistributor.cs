//------------------------------------------------------------------------------
// <copyright file="GridVectorDistributor.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    /// <summary>
    /// A grid vector distributor.
    /// </summary>
    public class GridVectorDistributor
    {
        private readonly List<List<GridVector>> grids;
        private readonly List<GridVector> vectors;

        /// <summary>
        /// Initializes a new instance of the <see cref="GridVectorDistributor"/> class.
        /// </summary>
        /// <param name="vectorStrings">The vector strings.</param>
        public GridVectorDistributor(List<string> vectorStrings)
        {
            this.vectors = this.ParseAndCalculateVectors(vectorStrings);
            this.grids = new List<List<GridVector>>();
        }

        /// <summary>
        /// Distribute vectors.
        /// </summary>
        public void DistributeVectors()
        {
            foreach (var vector in this.vectors)
            {
                this.PlaceVectorInGrid(vector);
            }
        }

        /// <summary>
        /// Place vector in grid.
        /// </summary>
        /// <param name="vector">The vector.</param>
        private void PlaceVectorInGrid(GridVector vector)
        {
            foreach (var grid in this.grids)
            {
                if (!grid.Any(v => v.Start == vector.Start))
                {
                    grid.Add(vector);
                    return;
                }
            }

            // If no suitable grid found, create a new one
            this.grids.Add(new List<GridVector> { vector });
        }

        /// <summary>
        /// Parse and calculate vectors.
        /// </summary>
        /// <param name="vectorStrings">The vector strings.</param>
        /// <returns>The parsed vectors.</returns>
        private List<GridVector> ParseAndCalculateVectors(List<string> vectorStrings)
        {
            var vectorsList = new List<GridVector>();
            foreach (var str in vectorStrings)
            {
                var startPiece = str.Substring(0, 2);
                var start = str.Substring(2, 2);
                var end = str.Substring(4, 2);
                var endPiece = str.Substring(6);
                var angle = this.CalculateAngle(start, end);
                vectorsList.Add(new GridVector(startPiece, endPiece, start, end, angle));
                vectorsList.Add(new GridVector(endPiece, startPiece, end, start, this.CalculateReverseAngle(angle))); // Reverse vector
            }

            return vectorsList;
        }

        /// <summary>
        /// Calculate angle.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <returns>The angle.</returns>
        private double CalculateAngle(string start, string end)
        {
            int startX = start[0] - 'a';
            int startY = start[1] - '1';
            int endX = end[0] - 'a';
            int endY = end[1] - '1';

            // Calculate differences in x and y coordinates
            int deltaX = endX - startX;
            int deltaY = endY - startY;

            // Calculate angle using arctan function, automatically handling the quadrant to return the angle in the range [-pi, pi]
            double angle = Math.Atan2(deltaY, deltaX);

            return angle;
        }

        /// <summary>
        /// Calculate reverse angle.
        /// </summary>
        /// <param name="originalAngle">The original angle.</param>
        /// <returns>The reverse angle.</returns>
        private double CalculateReverseAngle(double originalAngle)
        {
            // Add or subtract pi from the original angle based on its sign to get the reverse angle
            double reverseAngle = originalAngle >= 0 ? originalAngle - Math.PI : originalAngle + Math.PI;

            // Ensure the reverse angle stays within the range of -pi to pi
            if (reverseAngle > Math.PI)
            {
                reverseAngle -= 2 * Math.PI;
            }
            else if (reverseAngle <= -Math.PI)
            {
                reverseAngle += 2 * Math.PI;
            }

            return reverseAngle;
        }
    }
}
