//------------------------------------------------------------------------------
// <copyright file="BroadcastMapping.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// The mapping for broadcasting.
    /// </summary>
    public class BroadcastMapping
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BroadcastMapping"/> class.
        /// </summary>
        /// <param name="sourceIndicesA">The source indices A.</param>
        /// <param name="sourceIndicesB">The source indices B.</param>
        /// <param name="resultShape">The result shape.</param>
        /// <param name="reductionIndicesA">The reduction indices A.</param>
        /// <param name="reductionIndicesB">The reduction indices B.</param>
        public BroadcastMapping(
            int[] sourceIndicesA,
            int[] sourceIndicesB,
            int[] resultShape,
            int[] reductionIndicesA,
            int[] reductionIndicesB)
        {
            this.SourceIndicesA = sourceIndicesA;
            this.SourceIndicesB = sourceIndicesB;
            this.ResultShape = resultShape;
            this.ReductionIndicesA = reductionIndicesA;
            this.ReductionIndicesB = reductionIndicesB;
        }

        /// <summary>
        /// Gets the source indices A.
        /// </summary>
        public int[] SourceIndicesA { get; }

        /// <summary>
        /// Gets the source indices B.
        /// </summary>
        public int[] SourceIndicesB { get; }

        /// <summary>
        /// Gets the result shape.
        /// </summary>
        public int[] ResultShape { get; }

        /// <summary>
        /// Gets the reduction indices A.
        /// </summary>
        public int[] ReductionIndicesA { get; }

        /// <summary>
        /// Gets the reduction indices B.
        /// </summary>
        public int[] ReductionIndicesB { get; }
    }
}
