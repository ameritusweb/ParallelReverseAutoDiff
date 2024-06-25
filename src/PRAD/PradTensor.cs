//------------------------------------------------------------------------------
// <copyright file="PradTensor.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    /// <summary>
    /// A tensor and a PradOp.
    /// </summary>
    public class PradTensor : Tensor
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PradTensor"/> class.
        /// </summary>
        /// <param name="operation">The operation.</param>
        /// <param name="tensor">The tensor.</param>
        public PradTensor(PradOp operation, Tensor tensor)
            : base(tensor.Shape, tensor.Data)
        {
            this.PradOp = operation;
        }

        /// <summary>
        /// Gets a prad op.
        /// </summary>
        public PradOp PradOp { get; private set; }
    }
}
