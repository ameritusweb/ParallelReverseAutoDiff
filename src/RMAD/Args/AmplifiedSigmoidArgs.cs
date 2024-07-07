//------------------------------------------------------------------------------
// <copyright file="AmplifiedSigmoidArgs.cs" author="ameritusweb" date="7/6/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.RMAD
{
    using ParallelReverseAutoDiff.PRAD;

    /// <summary>
    /// Arguments for the amplified sigmoid operation.
    /// </summary>
    public class AmplifiedSigmoidArgs : IPradOperationArg<AmplifiedSigmoidOperation, Matrix>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="AmplifiedSigmoidArgs"/> class.
        /// </summary>
        /// <param name="m1">The first matrix.</param>
        public AmplifiedSigmoidArgs(Matrix m1)
        {
            this.M1 = m1;
        }

        /// <summary>
        /// Gets the first matrix.
        /// </summary>
        public Matrix M1 { get; }

        /// <summary>
        /// Gets the args.
        /// </summary>
        public Matrix Arg
        {
            get
            {
                return this.M1;
            }
        }
    }
}
