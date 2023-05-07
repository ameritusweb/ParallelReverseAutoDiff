//------------------------------------------------------------------------------
// <copyright file="NeuralNetwork.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    /// <summary>
    /// The base class for a neural network.
    /// </summary>
    public abstract class NeuralNetwork
    {
        private NeuralNetworkParameters parameters;

        /// <summary>
        /// Gets the parameters for the neural network.
        /// </summary>
        public NeuralNetworkParameters Parameters
        {
            get
            {
                return this.parameters ??= new NeuralNetworkParameters();
            }
        }
    }
}
