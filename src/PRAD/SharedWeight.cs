//------------------------------------------------------------------------------
// <copyright file="SharedWeight.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// A wrapper for shared weights that maintains a single tensor but allows multiple computational paths.
    /// </summary>
    public class SharedWeight
    {
        private Tensor weightTensor;
        private PradOp baseOp;
        private List<Guid> engineIds = new List<Guid>();
        private List<PradResult> endpoints = new List<PradResult>();

        /// <summary>
        /// Initializes a new instance of the <see cref="SharedWeight"/> class.
        /// </summary>
        /// <param name="tensor">The tensor to be shared.</param>
        public SharedWeight(Tensor tensor)
        {
            this.weightTensor = tensor;
            this.baseOp = new PradOp(tensor);
            this.baseOp.BackpropagationMode = BackpropagationMode.Accumulate;
            this.engineIds.Add(this.baseOp.EngineId);
        }

        /// <summary>
        /// Gets the shared PradOp.
        /// </summary>
        public PradOp SharedOp
        {
            get
            {
                return this.baseOp;
            }
        }

        /// <summary>
        /// Creates a new PradOp that shares the same underlying weight tensor.
        /// The gradients will automatically flow back to the original tensor during backpropagation.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>A PradOp that uses the shared weights.</returns>
        public PradOp UseOpAtIndex(int index)
        {
            if (index == 0)
            {
                return this.baseOp;
            }

            Guid engineId = Guid.NewGuid();
            this.engineIds.Add(engineId);
            this.baseOp.EngineId = engineId;
            return this.baseOp;
        }

        /// <summary>
        /// Registers an endpoint that uses this shared weight.
        /// </summary>
        /// <param name="endpoint">The endpoint result that uses this shared weight.</param>
        public void RegisterEndpoint(PradResult endpoint)
        {
            this.endpoints.Add(endpoint);
        }

        /// <summary>
        /// Backpropagate over the results.
        /// </summary>
        /// <param name="upstreamGradients">The upstream gradients.</param>
        public void BackpropagateAll(List<Tensor> upstreamGradients)
        {
            if (upstreamGradients.Count != this.endpoints.Count)
            {
                throw new System.ArgumentException("Number of gradients must match number of registered endpoints");
            }

            // Make sure gradients are reset before accumulation
            this.baseOp.ResetGradient();

            // Backpropagate through each endpoint with its corresponding engine ID
            for (int i = 0; i < this.endpoints.Count; ++i)
            {
                // Set the current engine ID to match the endpoint's context
                Guid engineId = this.engineIds[i];
                this.baseOp.EngineId = engineId;

                // Apply backpropagation for this endpoint
                this.endpoints[i].Back(upstreamGradients[i]);
            }
        }

        /// <summary>
        /// Resets both the gradients and the backpropagation steps for the next training iteration.
        /// </summary>
        public void Reset()
        {
            // Reset the base op
            this.baseOp.Reset();

            this.engineIds.Clear();
            this.engineIds.Add(this.baseOp.EngineId);
            this.endpoints.Clear();
        }

        /// <summary>
        /// Gets the current value of the shared weight tensor.
        /// </summary>
        /// <returns>The shared weight tensor.</returns>
        public Tensor GetTensor()
        {
            return this.weightTensor;
        }
    }
}
