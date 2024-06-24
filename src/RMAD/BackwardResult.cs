//------------------------------------------------------------------------------
// <copyright file="BackwardResult.cs" author="ameritusweb" date="5/15/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// The result of one step through a backward pass.
    /// </summary>
    public class BackwardResult
    {
        /// <summary>
        /// Gets or sets the backward pass gradients.
        /// </summary>
        public object?[] Results { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the result originated from an operation that has multiple inputs.
        /// </summary>
        public bool HasMultipleInputs { get; set; }

        /// <summary>
        /// Gets the first item.
        /// </summary>
        public object? Item1
        {
            get
            {
                return this.Results[0];
            }
        }

        /// <summary>
        /// Gets the second item.
        /// </summary>
        public object? Item2
        {
            get
            {
                return this.Results[1];
            }
        }

        /// <summary>
        /// Gets the third item.
        /// </summary>
        public object? Item3
        {
            get
            {
                return this.Results[2];
            }
        }
    }
}
