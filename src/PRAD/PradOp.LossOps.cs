//------------------------------------------------------------------------------
// <copyright file="PradOp.LossOps.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD.LossOps;

    /// <summary>
    /// The ops for the PradOp class.
    /// </summary>
    public partial class PradOp
    {
        /// <summary>
        /// Loss operation types.
        /// </summary>
        public class LossOps
        {
            /// <summary>
            /// Gets the Binary cross entropy loss op.
            /// </summary>
            public static Func<BinaryCrossEntropyLossOperation> BinaryCrossEntropyLossOp => () => new BinaryCrossEntropyLossOperation();

            /// <summary>
            /// Gets the Categorical cross entropy loss op.
            /// </summary>
            public static Func<CategoricalCrossEntropyLossOperation> CategoricalCrossEntropyLossOp => () => new CategoricalCrossEntropyLossOperation();

            /// <summary>
            /// Gets the Hober loss op.
            /// </summary>
            public static Func<HuberLossOperation> HuberLossOp => () => new HuberLossOperation();

            /// <summary>
            /// Gets the Mean absolute error loss op.
            /// </summary>
            public static Func<MeanAbsoluteErrorLossOperation> MeanAbsoluteErrorLossOp => () => new MeanAbsoluteErrorLossOperation();

            /// <summary>
            /// Gets the Mean squared error loss op.
            /// </summary>
            public static Func<MeanSquaredErrorLossOperation> MeanSquaredErrorLossOp => () => new MeanSquaredErrorLossOperation();
        }
    }
}
