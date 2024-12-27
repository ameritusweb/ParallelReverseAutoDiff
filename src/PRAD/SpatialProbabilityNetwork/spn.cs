//------------------------------------------------------------------------------
// <copyright file="spn.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.SpatialProbabilityNetwork
{
    using System;
    using System.Linq;

    /// <summary>
    /// Spatial probability network tools.
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("StyleCop.CSharp.NamingRules", "SA1300:Element should begin with upper-case letter", Justification = "To match TensorFlow")]
    public static class spn
    {
        /// <summary>
        /// Convert to a tensor.
        /// </summary>
        /// <param name="vector">The vector.</param>
        /// <returns>The tensor.</returns>
        /// <exception cref="System.NotImplementedException">Not supported.</exception>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor tensor(object vector)
        {
            if (vector is double[] doubleArray)
            {
                return new Tensor(new int[] { doubleArray.Length }, doubleArray);
            }

            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Multiple two tensors.
        /// </summary>
        /// <param name="leftTensor">The left tensor.</param>
        /// <param name="rightTensor">The right tensor.</param>
        /// <returns>The resultant tensor.</returns>
        /// <exception cref="System.NotImplementedException">Not supported.</exception>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor mul(object leftTensor, object rightTensor)
        {
            if (leftTensor is Tensor lTensor && rightTensor is Tensor rTensor)
            {
                return lTensor.ElementwiseMultiply(rTensor);
            }

            if (leftTensor is Tensor lt && rightTensor is double rDouble)
            {
                Tensor broadcasted = new Tensor(lt.Shape, rDouble);
                return lt.ElementwiseMultiply(broadcasted);
            }

            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Sum a tensor.
        /// </summary>
        /// <param name="tensor">The tensor to sum.</param>
        /// <returns>The sum.</returns>
        /// <exception cref="System.NotImplementedException">Not supported.</exception>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor sum(object tensor)
        {
            if (tensor is Tensor sTensor)
            {
                return sTensor.Sum(Enumerable.Range(0, sTensor.Shape.Length).ToArray());
            }

            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Sums the tensor along the specified axis.
        /// If axis is -1, sums across all elements.
        /// </summary>
        /// <param name="tensor">The tensor to sum.</param>
        /// <param name="axis">The axis along which to sum. Use -1 to sum all elements.</param>
        /// <returns>The summed tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor sum(object tensor, int axis = -1)
        {
            if (tensor is Tensor t)
            {
                if (axis == -1)
                {
                    return t.Sum(Enumerable.Range(0, t.Shape.Length).ToArray());
                }

                return t.Sum(new int[] { axis });
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Square a tensor.
        /// </summary>
        /// <param name="tensor">The tensor to square.</param>
        /// <returns>The square.</returns>
        /// <exception cref="System.NotImplementedException">Not supported.</exception>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor square(object tensor)
        {
            if (tensor is Tensor sTensor)
            {
                return sTensor.ElementwiseSquare();
            }

            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Create a random normal distribution.
        /// </summary>
        /// <param name="dimensions">The dimensions.</param>
        /// <returns>The distribution.</returns>
        /// <exception cref="System.NotImplementedException">Not supported.</exception>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor randomNormal(object dimensions)
        {
            if (dimensions is int[] dims)
            {
                return Tensor.RandomNormal(dims);
            }

            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Generates a tensor with values drawn from a uniform distribution between min and max.
        /// </summary>
        /// <param name="shape">Shape of the tensor.</param>
        /// <param name="min">Minimum value (default 0.0).</param>
        /// <param name="max">Maximum value (default 1.0).</param>
        /// <returns>Tensor with uniformly distributed random values.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor randomUniform(object shape, double min = 0.0, double max = 1.0)
        {
            if (shape is int[] dimensions)
            {
                return Tensor.RandomUniform(dimensions, min, max);
            }

            throw new NotImplementedException("Shape must be an integer array.");
        }

        /// <summary>
        /// Add two tensors element-wise.
        /// </summary>
        /// <param name="leftTensor">The left tensor.</param>
        /// <param name="rightTensor">The right tensor.</param>
        /// <returns>The resulting tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor add(object leftTensor, object rightTensor)
        {
            if (leftTensor is Tensor lTensor && rightTensor is Tensor rTensor)
            {
                return lTensor.ElementwiseAdd(rTensor);
            }

            throw new System.NotImplementedException("Inputs must be tensors.");
        }

        /// <summary>
        /// Divide one tensor by another element-wise.
        /// </summary>
        /// <param name="leftTensor">The left tensor (numerator).</param>
        /// <param name="rightTensor">The right tensor (denominator).</param>
        /// <returns>The resulting tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor div(object leftTensor, object rightTensor)
        {
            if (leftTensor is Tensor lTensor && rightTensor is Tensor rTensor)
            {
                return lTensor.ElementwiseDivide(rTensor);
            }

            throw new System.NotImplementedException("Inputs must be tensors.");
        }

        /// <summary>
        /// Compute the norm (magnitude) of a tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The norm as a scalar tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor norm(object tensor)
        {
            if (tensor is Tensor sTensor)
            {
                var squared = sTensor.ElementwiseSquare();
                var summed = squared.Sum(Enumerable.Range(0, sTensor.Shape.Length).ToArray());
                return summed.ElementwiseSquareRoot();
            }

            throw new System.NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Calculate the volatility.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>The volatility.</returns>
        /// <exception cref="NotImplementedException">Not supported.</exception>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor volatility(object tensor)
        {
            if (tensor is Tensor t)
            {
                var (mean, variance) = t.Moments();
                return spn.div(variance, spn.add(mean, 1e-8));
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Expand the dimensions of a tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="axis">The axis.</param>
        /// <returns>The expanded tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor expandDims(object tensor, int axis)
        {
            if (tensor is Tensor t)
            {
                return t.ExpandDims(axis);
            }

            throw new NotImplementedException("Invalid inputs for expandDims.");
        }

        /// <summary>
        /// Stack tensors along a new axis.
        /// </summary>
        /// <param name="tensors">Array of tensors to stack.</param>
        /// <returns>Stacked tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor stack(object tensors)
        {
            if (tensors is Tensor[] ts)
            {
                if (ts.Length == 0)
                {
                    throw new ArgumentException("Tensor array cannot be empty.");
                }

                return Tensor.Stack(ts);
            }

            throw new NotImplementedException("Input must be a tensor array.");
        }

        /// <summary>
        /// Compute the mean along a specified axis.
        /// </summary>
        /// <param name="tensor">Input tensor.</param>
        /// <param name="axis">Axis along which to compute the mean.</param>
        /// <returns>Tensor representing the mean.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor mean(object tensor, int axis)
        {
            if (tensor is Tensor t)
            {
                var summed = t.Sum(new int[] { axis });
                var count = t.Shape[axis];
                return summed.Divide(count);
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Computes the mean of all elements in the tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The mean as a scalar tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor mean(object tensor)
        {
            if (tensor is Tensor t)
            {
                var summed = spn.sum(t);
                var count = t.Shape.Aggregate((a, b) => a * b);  // Total number of elements
                return spn.div(summed, count);
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Computes the mean and variance of the tensor along a specified axis.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="axis">Axis along which to compute moments.</param>
        /// <returns>A tuple containing the mean and variance tensors.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static (Tensor Mean, Tensor Variance) moments(object tensor, int axis = -1)
        {
            if (tensor is Tensor t)
            {
                // Compute mean
                var mean = t.Mean(axis);

                // Subtract mean and square
                var diff = t.ElementwiseSub(mean);
                var squaredDiff = diff.ElementwiseSquare();

                // Compute variance (mean of squared differences)
                var variance = squaredDiff.Mean(axis);

                return (mean, variance);
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Matrix multiplication of two tensors.
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns>The resultant tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor matmul(object a, object b)
        {
            if (a is Tensor t1 && b is Tensor t2)
            {
                return t1.MatrixMultiply(t2);
            }

            throw new NotImplementedException("Inputs must be tensors for matmul.");
        }

        /// <summary>
        /// Element-wise logarithm of a tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The resultant tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor log(object tensor)
        {
            if (tensor is Tensor t)
            {
                return t.Log();
            }

            throw new NotImplementedException("Input must be a tensor for log.");
        }

        /// <summary>
        /// Element-wise exponential of a tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The resultant tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor exp(object tensor)
        {
            if (tensor is Tensor t)
            {
                return t.Exp();
            }

            throw new NotImplementedException("Input must be a tensor for exp.");
        }

        /// <summary>
        /// Apply softmax to a tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The resultant tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor softmax(object tensor)
        {
            if (tensor is Tensor t)
            {
                return t.Softmax();
            }

            throw new NotImplementedException("Input must be a tensor for softmax.");
        }

        /// <summary>
        /// Subtract one tensor from another element-wise.
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns>The resultant tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor sub(object a, object b)
        {
            if (a is Tensor t1 && b is Tensor t2)
            {
                return t1.ElementwiseSub(t2);
            }

            throw new NotImplementedException("Inputs must be tensors for sub.");
        }

        /// <summary>
        /// Create a range tensor from start to end.
        /// </summary>
        /// <param name="start">The start.</param>
        /// <param name="end">The end.</param>
        /// <returns>The resultant tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor range(int start, int end)
        {
            if (start is int s && end is int e)
            {
                return Tensor.Range(s, e);
            }

            throw new NotImplementedException("Inputs must be integers for range.");
        }

        /// <summary>
        /// Get the tensor data as a flat array.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>The tensor data.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static double[] DataSync(object tensor)
        {
            if (tensor is Tensor t)
            {
                return t.Data;
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Computes the dot product of two tensors.
        /// </summary>
        /// <param name="a">The first tensor.</param>
        /// <param name="b">The second tensor.</param>
        /// <returns>The dot product as a scalar tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor dot(object a, object b)
        {
            if (a is Tensor t1 && b is Tensor t2)
            {
                // Perform element-wise multiplication
                var multiplied = spn.mul(t1, t2);

                // Sum along all axes to get the scalar dot product
                var result = spn.sum(multiplied);

                return result;
            }

            throw new NotImplementedException("Inputs must be tensors for dot.");
        }

        /// <summary>
        /// Compares two tensors element-wise and returns a mask where the left tensor is greater than the right tensor.
        /// </summary>
        /// <param name="leftTensor">The left tensor.</param>
        /// <param name="rightTensor">The right tensor.</param>
        /// <returns>A tensor with 1.0 where the left tensor is greater, 0.0 otherwise.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor greater(object leftTensor, object rightTensor)
        {
            if (leftTensor is Tensor lTensor && rightTensor is Tensor rTensor)
            {
                return lTensor.GreaterThan(rTensor);
            }

            if (leftTensor is Tensor lt && rightTensor is double rDouble)
            {
                Tensor broadcasted = new Tensor(lt.Shape, rDouble);
                return lt.GreaterThan(broadcasted);
            }

            throw new NotImplementedException("Inputs must be tensors for greater.");
        }

        /// <summary>
        /// Computes the element-wise negation of a tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The negated tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor neg(object tensor)
        {
            if (tensor is Tensor t)
            {
                return t.ElementwiseMultiply(new Tensor(t.Shape, -1.0d));
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Computes the standard deviation of a tensor along the specified axis.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="axis">The axis along which to compute the standard deviation. Use -1 to compute over all elements.</param>
        /// <returns>The standard deviation tensor.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor std(object tensor, int axis = -1)
        {
            if (tensor is Tensor t)
            {
                var (mean, variance) = spn.moments(t, axis);
                return spn.sqrt(variance);  // std = sqrt(variance)
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// Computes the element-wise square root of a tensor.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <returns>The tensor with square root applied element-wise.</returns>
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006:Naming Styles", Justification = "To match TensorFlow")]
        public static Tensor sqrt(object tensor)
        {
            if (tensor is Tensor t)
            {
                return t.ElementwiseSquareRoot();
            }

            throw new NotImplementedException("Input must be a tensor.");
        }

        /// <summary>
        /// A map.
        /// </summary>
        /// <returns>The map.</returns>
        public static JsMap Map()
        {
            return new JsMap();
        }

        /// <summary>
        /// A set.
        /// </summary>
        /// <returns>The set.</returns>
        public static JsSet Set()
        {
            return new JsSet();
        }
    }
}
