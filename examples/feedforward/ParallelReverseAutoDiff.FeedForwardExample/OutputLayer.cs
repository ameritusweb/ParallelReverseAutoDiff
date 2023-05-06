//------------------------------------------------------------------------------
// <copyright file="OutputLayer.cs" author="ameritusweb" date="5/5/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.FeedForwardExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The output layer.
    /// </summary>
    public class OutputLayer
    {
        /// <summary>
        /// Gets or sets the weight matrix for the hidden layer.
        /// </summary>
        public Matrix V { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the hidden layer.
        /// </summary>
        public Matrix Bo { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the weight matrix with respect to the loss function.
        /// </summary>
        public Matrix DV { get; set; }

        /// <summary>
        /// Gets or sets the gradient of the bias matrix with respect to the loss function.
        /// </summary>
        public Matrix DBo { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MV { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the weight matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VV { get; set; }

        /// <summary>
        /// Gets or sets the first moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix MBo { get; set; }

        /// <summary>
        /// Gets or sets the second moment (moving average) of the bias matrix's gradients, used in optimization algorithms like Adam.
        /// </summary>
        public Matrix VBo { get; set; }
    }
}
