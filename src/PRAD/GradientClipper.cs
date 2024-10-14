//------------------------------------------------------------------------------
// <copyright file="GradientClipper.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Linq;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Clips gradients.
    /// </summary>
    public class GradientClipper : IClipper
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GradientClipper"/> class.
        /// </summary>
        /// <param name="clipValue">The clip value.</param>
        public GradientClipper(double clipValue = 4)
        {
            this.ClipValue = clipValue;
        }

        /// <summary>
        /// Gets or sets the clip value.
        /// </summary>
        public double ClipValue { get; set; }

        /// <summary>
        /// Clips the gradients of the provided tensor based on a clip value and a minimum threshold.
        /// </summary>
        /// <param name="gradients">The tensor of gradients to clip.</param>
        /// <returns>The clipped tensor.</returns>
        public Tensor ClipGradients(Tensor gradients)
        {
            // Step 1: Standardize the gradients
            var standardizedGradients = this.StandardizedTensor(gradients);

            // Step 2: Compute the dynamic clip value element-wise: Math.Min(clipValue, 1 + |standardizedGradients|)
            var absStandardizedGradients = standardizedGradients.Abs();  // |standardizedGradients|
            var dynamicClip = absStandardizedGradients.ElementwiseAdd(new Tensor(absStandardizedGradients.Shape, 1));  // 1 + |standardizedGradients|
            dynamicClip = dynamicClip.Min(new Tensor(dynamicClip.Shape, PradTools.Cast(this.ClipValue)));  // Clip with Math.Min(clipValue, ...)

            // Step 3: Clip the gradient values within [-dynamicClip, dynamicClip]
            var clippedGradients = gradients.Min(dynamicClip).Max(dynamicClip.ElementwiseNegate());

            return clippedGradients;
        }

        /// <summary>
        /// Standardizes the tensor (subtracts mean, divides by standard deviation).
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The standardized tensor.</returns>
        public Tensor StandardizedTensor(Tensor tensor)
        {
            // Step 1: Compute the mean of the tensor.
            double mean = tensor.Data.Average();

            // Step 2: Compute the standard deviation of the tensor.
            double variance = tensor.Data.Select(x => Math.Pow(x - mean, 2)).Average();
            double stdDev = Math.Sqrt(variance);

            // Step 3: Standardize the tensor (element-wise operation).
            Tensor standardizedTensor = tensor.ElementwiseSub(new Tensor(tensor.Shape, PradTools.Cast(mean)))
                                            .ElementwiseDivide(new Tensor(tensor.Shape, PradTools.Cast(stdDev)));

            return standardizedTensor;
        }
    }
}
