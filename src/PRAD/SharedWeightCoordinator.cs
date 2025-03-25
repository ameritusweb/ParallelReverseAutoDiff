//------------------------------------------------------------------------------
// <copyright file="SharedWeightCoordinator.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System.Collections.Generic;

    /// <summary>
    /// A coordinator for managing backpropagation across multiple shared weights.
    /// </summary>
    public class SharedWeightCoordinator
    {
        private List<SharedWeight> sharedWeights = new List<SharedWeight>();

        /// <summary>
        /// Registers a shared weight with the coordinator.
        /// </summary>
        /// <param name="sharedWeight">The shared weight to register.</param>
        public void RegisterSharedWeight(SharedWeight sharedWeight)
        {
            this.sharedWeights.Add(sharedWeight);
        }

        /// <summary>
        /// Registers a result with all shared weights.
        /// </summary>
        /// <param name="result">The result to register.</param>
        public void RegisterResult(PradResult result)
        {
            // Register with all shared weights
            foreach (var sharedWeight in this.sharedWeights)
            {
                sharedWeight.RegisterEndpoint(result);
            }
        }

        /// <summary>
        /// Performs backpropagation through all registered results.
        /// </summary>
        /// <param name="upstreamGradients">The upstream gradients for each result.</param>
        public void BackpropagateAll(List<Tensor> upstreamGradients)
        {
            if (this.sharedWeights.Count > 0)
            {
                this.sharedWeights[0].BackpropagateAll(upstreamGradients);
            }
        }

        /// <summary>
        /// Resets all shared weights for the next training iteration.
        /// </summary>
        public void Reset()
        {
            foreach (var sharedWeight in this.sharedWeights)
            {
                sharedWeight.Reset();
            }
        }
    }
}
