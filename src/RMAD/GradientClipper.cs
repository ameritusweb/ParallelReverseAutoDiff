//------------------------------------------------------------------------------
// <copyright file="GradientClipper.cs" author="ameritusweb" date="5/7/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    using System.Collections.Generic;

    /// <summary>
    /// A gradient clipper.
    /// </summary>
    public class GradientClipper
    {
        private NeuralNetwork network;

        /// <summary>
        /// Initializes a new instance of the <see cref="GradientClipper"/> class.
        /// </summary>
        /// <param name="network">The neural network.</param>
        public GradientClipper(NeuralNetwork network)
        {
            this.network = network;
        }
    }
}
