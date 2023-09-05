// ------------------------------------------------------------------------------
// <copyright file="Point3d.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.FsmnnExample.Amaze
{
    /// <summary>
    /// Provides point data.
    /// </summary>
    public class Point3d
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Point3d"/> class.
        /// </summary>
        /// <param name="x">X.</param>
        /// <param name="y">Y.</param>
        /// <param name="z">Z.</param>
        public Point3d(double x, double y, double z)
        {
            this.X = x;
            this.Y = y;
            this.Z = z;
        }

        /// <summary>
        /// Gets or sets X.
        /// </summary>
        public double X { get; set; }

        /// <summary>
        /// Gets or sets Y.
        /// </summary>
        public double Y { get; set; }

        /// <summary>
        /// Gets or sets Z.
        /// </summary>
        public double Z { get; set; }
    }
}
