//------------------------------------------------------------------------------
// <copyright file="PradVectorTools.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// PRAD Tools for Vector Neural Networks.
    /// </summary>
    public class PradVectorTools
    {
        /// <summary>
        /// Vectorize the input with the angles.
        /// </summary>
        /// <param name="input">The input prad op.</param>
        /// <param name="angles">The angles prad op.</param>
        /// <returns>The vectorized result.</returns>
        public PradResult Vectorize(PradOp input, PradOp angles)
        {
            var map = new Dictionary<(double, double, double), int>();

            Func<PradOp, PradOp, int, object, int> func =
                (a, b, index, mappingObj) =>
                {
                    var bTensor = b.SeedResult.Result;
                    var mapping = (Dictionary<(double, double, double), int>)mappingObj;
                    var (i, j) = (index / bTensor.Shape[1], index % bTensor.Shape[1]);
                    double prev = j > 0 ? bTensor[i, j - 1] : double.MinValue;
                    double current = bTensor[i, j];
                    double next = j < bTensor.Shape[1] - 1 ? bTensor[i, j + 1] : double.MaxValue;
                    var context = (prev, current, next);

                    if (!mapping.ContainsKey(context))
                    {
                        mapping[context] = mapping.Count;
                    }

                    return mapping[context];
                };

            var result = this.ConcatMap(
                input,
                angles,
                func,
                map);

            return result;
        }

        private PradResult ConcatMap(
            PradOp b,
            PradOp a,
            Func<PradOp, PradOp, int, object, int> mapFunc,
            object mappingObj)
        {
            var flattenedA = a.Reshape(new[] { a.CurrentShape[0] * a.CurrentShape[1] });

            var indices = new Tensor(b.CurrentShape);
            for (int i = 0; i < b.CurrentShape[0]; i++)
            {
                for (int j = 0; j < b.CurrentShape[1]; j++)
                {
                    int index = (i * b.CurrentShape[1]) + j;
                    indices[i, j] = mapFunc(a, b, index, mappingObj);
                }
            }

            var resortedFlatA = flattenedA.PradOp.Gather(indices);
            var resortedA = resortedFlatA.PradOp.Reshape(a.CurrentShape);

            return b.Concat(new[] { resortedA.Result }, axis: 1);
        }
    }
}
