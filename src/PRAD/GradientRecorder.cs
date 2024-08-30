//------------------------------------------------------------------------------
// <copyright file="GradientRecorder.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// A singleton class to record gradients in order.
    /// </summary>
    public class GradientRecorder
    {
        // Singleton instance
        private static readonly Lazy<GradientRecorder> ReadonlyInstance = new Lazy<GradientRecorder>(() => new GradientRecorder());

        // List to store operations and their associated gradients in order
        private readonly List<(string OperationName, string Gradient)> recordedGradients;

        // Private constructor to prevent external instantiation
        private GradientRecorder()
        {
            this.recordedGradients = new List<(string OperationName, string Gradient)>();
        }

        /// <summary>
        /// Gets the gradient recorder instance.
        /// </summary>
        public static GradientRecorder Instance => ReadonlyInstance.Value;

        /// <summary>
        /// Gets or sets a value indicating whether recording is enabled.
        /// </summary>
        public bool RecordingEnabled { get; set; } = false;

        /// <summary>
        /// Records the gradient of an operation.
        /// </summary>
        /// <param name="operationName">The operation name.</param>
        /// <param name="gradients">The gradients.</param>
        public void RecordGradient(string operationName, Tensor[]? gradients)
        {
            if (!this.RecordingEnabled)
            {
                return;
            }

            if (gradients == null)
            {
                this.recordedGradients.Add((operationName, string.Empty));
            }

            this.recordedGradients.Add((operationName, string.Join("\n\n", gradients.Select(x => x.PrintCode(4)))));
        }

        /// <summary>
        /// Retrieves the recorded gradients.
        /// </summary>
        /// <returns>The recorded gradients.</returns>
        public List<(string OperationName, string Gradient)> GetRecordedGradients()
        {
            return new List<(string OperationName, string Gradient)>(this.recordedGradients);
        }

        /// <summary>
        /// Clear the recorded gradients.
        /// </summary>
        public void Clear()
        {
            this.recordedGradients.Clear();
        }
    }
}
