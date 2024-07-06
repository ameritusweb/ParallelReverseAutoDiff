//------------------------------------------------------------------------------
// <copyright file="LossOps.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Loss operation types.
    /// </summary>
    public class LossOps
    {
        /// <summary>
        /// Gets the Binary cross entropy loss op type.
        /// </summary>
        public static Type BinaryCrossEntropyLossOpType => typeof(BinaryCrossEntropyLossOperation);

        /// <summary>
        /// Gets the Categorical cross entropy loss op type.
        /// </summary>
        public static Type CategoricalCrossEntropyLossOpType => typeof(CategoricalCrossEntropyLossOperation);

        /// <summary>
        /// Gets the Hober loss op type.
        /// </summary>
        public static Type HuberLossOpType => typeof(HuberLossOperation);

        /// <summary>
        /// Gets the Mean absolute error loss op type.
        /// </summary>
        public static Type MeanAbsoluteErrorLossOpType => typeof(MeanAbsoluteErrorLossOperation);

        /// <summary>
        /// Gets the Mean squared error loss op type.
        /// </summary>
        public static Type MeanSquaredErrorLossOpType => typeof(MeanSquaredErrorLossOperation);
    }
}
