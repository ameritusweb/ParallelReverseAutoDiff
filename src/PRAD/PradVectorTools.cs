//------------------------------------------------------------------------------
// <copyright file="PradVectorTools.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// PRAD Tools for Vector Neural Networks.
    /// </summary>
    public class PradVectorTools
    {
        /// <summary>
        /// Uses element-wise square on inputs or angles.
        /// </summary>
        /// <param name="input">The input to square.</param>
        /// <returns>The squared result.</returns>
        public PradResult ElementwiseSquare(PradOp input)
        {
            return input.Square();
        }

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

        /// <summary>
        /// Perform vector decomposition that follows the formula: I + O = (Ai + Bi) * W.
        /// </summary>
        /// <param name="opInput1">The first input.</param>
        /// <param name="opInput2">The second input.</param>
        /// <param name="opWeights">The weights.</param>
        /// <returns>The result of the decomposition.</returns>
        public PradResult VectorDecomposition(PradOp opInput1, PradOp opInput2, PradOp opWeights)
        {
            var num_rows = opInput1.SeedResult.Result.Shape[0];
            var num_cols = opInput1.SeedResult.Result.Shape[1] / 2;
            var size = num_rows * num_cols;

            var (magnitude, angle) = opInput1.DoParallel(
                x => x.Indexer(":", $":{num_cols}"),
                y => y.Indexer(":", $"{num_cols}:"));
            var magnitudeBranch = magnitude.Branch();
            var angleBranch = angle.Branch();

            var input2_cols = opInput2.SeedResult.Result.Shape[1];
            var half_cols = input2_cols / 2;

            var opInput2Branch = opInput2.Branch();

            var opInput2Branches = opInput2.BranchStack(3);

            var w_magnitudes_t = new PradResult[4];
            var w_angles_t = new PradResult[4];
            for (int i = 0; i < 4; i++)
            {
                var branchM = opInput2;

                if (i > 0)
                {
                    branchM = opInput2Branches.Pop();
                }

                var (w_magnitudes_tt, w_angles_tt) = branchM.DoParallel(
                    x => x.Indexer(":", $"{1 + i}:{half_cols}:5"),
                    y => y.Indexer(":", $"{half_cols + 1 + i}::5"));

                w_magnitudes_t[i] = w_magnitudes_tt;
                w_angles_t[i] = w_angles_tt;
            }

            var w_magnitudes_stacked = w_magnitudes_t[0].PradOp.Stack(w_magnitudes_t.Select(f => f.Result).Skip(1).ToArray(), axis: -1);
            var w_angles_stacked = w_angles_t[0].PradOp.Stack(w_angles_t.Select(f => f.Result).Skip(1).ToArray(), axis: -1);

            var w_magnitudes = w_magnitudes_stacked;
            var w_angles = w_angles_stacked;
            var w_magnitudesBranch = w_magnitudes.Branch();
            var w_anglesBranch = w_angles.Branch();

            var w_magnitudesTrans = w_magnitudesBranch.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });
            var w_anglesTrans = w_anglesBranch.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            var (w_magnitude_pivot_a, w_angle_pivot_a) = opInput2Branch.DoParallel(
                x => x.Indexer(":", $":{half_cols}"),
                y => y.Indexer(":", $"{half_cols}:"));

            var w_magnitude_pivot = w_magnitude_pivot_a.PradOp.Indexer(":", $"::5");
            var w_angle_pivot = w_angle_pivot_a.PradOp.Indexer(":", $"::5");

            var w_magnitude_pivotBranch = w_magnitude_pivot.Branch();
            var w_angle_pivotBranch = w_angle_pivot.Branch();

            var (cosAngles, sinAngles) = angle.PradOp.DoParallel(
                x => x.Cos(),
                y => y.Sin());

            var (x, y) = magnitude.PradOp.DoParallel(
                x => x.Mul(cosAngles.Result),
                x => x.Mul(sinAngles.Result));

            var xResult = x.Result;
            var yResult = y.Result;

            var (cosPivot, sinPivot) = w_angle_pivot.PradOp.DoParallel(
                x => x.Cos(),
                x => x.Sin());

            var (xPivot, yPivot) = w_magnitude_pivot.PradOp.DoParallel(
                x => x.Mul(cosPivot.Result),
                x => x.Mul(sinPivot.Result));

            var xPivotResult = xPivot.Result;
            var yPivotResult = yPivot.Result;

            var (cosAngles_w, sinAngles_w) = w_angles.PradOp.DoParallel(
                x => x.Cos(),
                x => x.Sin());

            var (x_w, y_w) = w_magnitudes.PradOp.DoParallel(
                x => x.Mul(cosAngles_w.Result),
                x => x.Mul(sinAngles_w.Result));

            Tensor addScalar = new Tensor(opWeights.SeedResult.Result.Shape, 0.01d);
            var adjustedWeights = opWeights.Add(addScalar);

            var xPivotAdd = x.Then(PradOp.AddOp, xPivot.Result);
            var yPivotAdd = y.Then(PradOp.AddOp, yPivot.Result);

            var weightsEpsilon = adjustedWeights.Then(PradOp.AddOp, new Tensor(opWeights.SeedResult.Result.Shape, 1e-9d));

            var sumX = xPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);
            var sumY = yPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);

            var sumXExpanded = sumX.Then(PradOp.ExpandDimsOp, -1);
            var sumYExpanded = sumY.Then(PradOp.ExpandDimsOp, -1);

            var sumXBranch = sumX.Branch();
            var negativeSumXExpanded = sumXBranch.Mul(new Tensor(sumXBranch.CurrentShape, -1d));

            var sumYBranch = sumY.Branch();
            var negativeSumYExpanded = sumYBranch.Mul(new Tensor(sumYBranch.CurrentShape, -1d));

            var sumXReshaped = sumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumXReshaped = negativeSumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            var x_wReshaped = x_w.Then(PradOp.ReshapeOp, new int[] { 4, size });

            var sumXReshapedBranch = sumXReshaped.Branch();

            var sumXConcatenated = sumXReshaped.Then(
                PradOp.ConcatOp,
                new[] { negativeSumXReshaped.Result, sumXReshapedBranch.BranchInitialTensor, negativeSumXReshaped.Result },
                axis: 0);

            var diffX = x_wReshaped.PradOp.SubFrom(sumXConcatenated.Result);

            var sumYReshaped = sumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumYReshaped = negativeSumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            var y_wReshaped = y_w.Then(PradOp.ReshapeOp, new int[] { 4, size });

            var sumYReshapedBranch = sumYReshaped.Branch();

            var sumYConcatenated = sumYReshaped.Then(
                PradOp.ConcatOp,
                new[] { negativeSumYReshaped.Result, sumYReshapedBranch.BranchInitialTensor, negativeSumYReshaped.Result },
                axis: 0);

            var diffY = sumYConcatenated.Then(PradOp.SubOp, y_wReshaped.Result);

            var diffXBranch = diffX.Branch();
            var diffYBranch = diffY.Branch();

            var diffXSquared = diffX.Then(PradOp.SquareOp);
            var diffYSquared = diffY.Then(PradOp.SquareOp);

            var resultMagnitudes = diffXSquared.Then(PradOp.AddOp, diffYSquared.Result)
                                                .Then(PradOp.SquareRootOp);

            var resultAngles = diffYBranch.Atan2(diffXBranch.BranchInitialTensor);

            var resultMagnitudesTrans2 = resultMagnitudes.PradOp.Reshape(1, 4, -1);
            var resultMagnitudesTrans = resultMagnitudesTrans2.PradOp.Transpose(new int[] { 0, 2, 1 });
            var resultAnglesTrans = resultAngles.PradOp.Reshape(1, 4, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            var reshapedResultMagnitudes = resultMagnitudesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            var reshapedResultAngles = resultAnglesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            var magnitudeReshaped = magnitudeBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var angleReshaped = angleBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var w_magnitude_pivotReshaped = w_magnitude_pivotBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var w_angle_pivotReshaped = w_angle_pivotBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var w_magnitudesReshaped = w_magnitudesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 4 });
            var w_anglesReshaped = w_anglesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 4 });
            var resultMagnitudesReshaped = reshapedResultMagnitudes.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 4 });
            var resultAnglesReshaped = reshapedResultAngles.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 4 });

            var magnitudesPart = resultMagnitudesReshaped.Then(
                PradOp.ConcatOp,
                new[] { magnitudeReshaped.Result, w_magnitude_pivotReshaped.Result, w_magnitudesReshaped.Result },
                axis: 2,
                new int[] { 1, 2, 3, 0 });

            var anglesPart = angleReshaped.Then(
                PradOp.ConcatOp,
                new[] { w_angle_pivotReshaped.Result, w_anglesReshaped.Result, resultAnglesReshaped.Result },
                axis: 2);

            var output = magnitudesPart.Then(PradOp.ConcatOp, new[] { anglesPart.Result }, axis: 1);

            var finalOutput = output.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols * 20 });

            return finalOutput;
        }

        /// <summary>
        /// Performs a vector-based matrix multiplication.
        /// </summary>
        /// <param name="opInput1">The first input.</param>
        /// <param name="opInput2">The second input.</param>
        /// <param name="opWeights">The weights.</param>
        /// <returns>The result of the vector-based matrix multiplication.</returns>
        public PradResult VectorBasedMatrixMultiplication(PradOp opInput1, PradOp opInput2, PradOp opWeights)
        {
            var clonedOpInput1 = opInput1.Branch();
            var clonedOpInput2 = opInput2.Branch();

            var rows = opInput1.SeedResult.Result.Shape[0];
            var cols = opInput1.SeedResult.Result.Shape[1];
            var halfCols = cols / 2;

            var anglesSeed = opInput1.Indexer(":", $"{halfCols}:").Result;
            var anglesOther = opInput2.Indexer(":", $"{halfCols}:").Result;

            var concatAngles = opInput1.Concat(new[] { anglesOther }, axis: 1).Result;

            var flatAngles = opInput1.Reshape(new int[] { 1, -1 }).Result;
            var flatAnglesOp = opInput1.Branch();

            var sinAngles = opInput1.Sin();
            var cosAngles = flatAnglesOp.Cos().Result;

            var magnitudesSeed = clonedOpInput1.Indexer(":", $":{halfCols}").Result;
            var magnitudesOther = clonedOpInput2.Indexer(":", $":{halfCols}").Result;

            var concatMagnitudes = clonedOpInput1.Concat(new[] { magnitudesOther }, axis: 1).Result;

            var flatMagnitudes = clonedOpInput1.Reshape(new int[] { 1, -1 }).Result;
            var flatMagnitudesOp = clonedOpInput1.Branch();

            var ys = sinAngles.PradOp.Mul(flatMagnitudes);
            var xs = flatMagnitudesOp.Mul(cosAngles).Result;

            var reshapedYs = ys.PradOp.Reshape(new int[] { rows, cols });
            var reshapedXs = flatMagnitudesOp.Reshape(new int[] { rows, cols }).Result;
            var reshapedYsOp = ys.PradOp.Branch();
            var reshapedXsOp = flatMagnitudesOp.Branch();

            var y1s = reshapedYs.PradOp.Indexer(":", $":{halfCols}");
            var y2s = reshapedYsOp.Indexer(":", $"{halfCols}:").Result;
            var x1s = flatMagnitudesOp.Indexer(":", $":{halfCols}").Result;
            var x2s = reshapedXsOp.Indexer(":", $"{halfCols}:").Result;

            var columnSize = rows * rows * halfCols;

            // For X2 and Y2
            var x2Reshaped = reshapedXsOp.Transpose(new int[] { 1, 0 }).Result;
            var x2Tiled = reshapedXsOp.Tile(new int[] { rows, 1 }).Result;
            var flatX2s = reshapedXsOp.Reshape(new int[] { 1, columnSize }).Result;

            var y2Reshaped = reshapedYsOp.Transpose(new int[] { 1, 0 }).Result;
            var y2Tiled = reshapedYsOp.Tile(new int[] { rows, 1 });
            var flatY2s = reshapedYsOp.Reshape(new int[] { 1, columnSize });

            // For X1 and Y1
            var x1Tiled = flatMagnitudesOp.Tile(new int[] { 1, rows }).Result;
            var flatX1s = flatMagnitudesOp.Reshape(new int[] { 1, columnSize });

            var y1Tiled = reshapedYs.PradOp.Tile(new int[] { 1, rows }).Result;
            var flatY1s = reshapedYs.PradOp.Reshape(new int[] { 1, columnSize });

            var deltaY = flatY1s.PradOp.SubFrom(reshapedYsOp.Result!);
            var deltaX = flatX1s.PradOp.SubFrom(reshapedXsOp.Result!);
            var deltaYOp = deltaY.Branch();

            var squaredY = deltaY.PradOp.Square();
            var squaredX = deltaX.PradOp.Square().Result;

            var addedYX = squaredY.PradOp.Add(squaredX);

            var unweightedMagnitude = addedYX.PradOp.SquareRoot();

            var transposedWeights = opWeights.Transpose(new int[] { 1, 0 }).Result;
            var tiledWeights = opWeights.Tile(new int[] { rows, 1 }).Result;
            var flattenedWeights = opWeights.Reshape(new int[] { 1, -1 }).Result;

            var magnitudes = unweightedMagnitude.PradOp.Mul(flattenedWeights);
            var magnitudesOp = magnitudes.Branch();

            var angles = deltaYOp.Atan2(deltaX.Result).Result;
            var anglesOp = deltaYOp.Branch();

            var sinAngles2 = deltaYOp.Sin().Result;
            var cosAngles2 = anglesOp.Cos().Result;

            var yOverall = magnitudes.PradOp.Mul(sinAngles2);
            var xOverall = magnitudesOp.Mul(cosAngles2);

            var reshapedYOverall = yOverall.PradOp.Reshape(new int[] { rows * halfCols, rows });
            var reshapedXOverall = xOverall.PradOp.Reshape(new int[] { rows * halfCols, rows });

            var sumRowsY = reshapedYOverall.PradOp.SumRows();
            var sumRowsX = reshapedXOverall.PradOp.SumRows();

            var flattenedSumRowsY = sumRowsY.PradOp.Reshape(new int[] { 1, -1 });
            var flattenedSumRowsX = sumRowsX.PradOp.Reshape(new int[] { 1, -1 });
            var flattenedSumRowsYOp = flattenedSumRowsY.Branch();
            var flattenedSumRowsXOp = flattenedSumRowsX.Branch();

            var flattenedYSquared = flattenedSumRowsY.PradOp.Square();
            var flattenedXSquared = flattenedSumRowsX.PradOp.Square();

            var addedYXOverall = flattenedYSquared.PradOp.Add(flattenedXSquared.Result);

            var magnitudesOverall = addedYXOverall.PradOp.SquareRoot();
            var anglesOverall = flattenedSumRowsYOp.Atan2(flattenedSumRowsXOp.BranchInitialTensor);

            var reshapedMagnitudesOverall = magnitudesOverall.PradOp.Reshape(new int[] { rows, halfCols });
            var reshapedAnglesOverall = anglesOverall.PradOp.Reshape(new int[] { rows, halfCols }).Result;

            var outputTensor = reshapedMagnitudesOverall.PradOp.Concat(new[] { reshapedAnglesOverall }, axis: 1);
            var o = outputTensor.PradOp.Reshape(new int[] { rows, cols });

            return o;
        }

        /// <summary>
        /// Perform a vector weighted add operation.
        /// </summary>
        /// <param name="opInput1">The first input.</param>
        /// <param name="opInput2">The second input.</param>
        /// <param name="opWeights">The weights.</param>
        /// <returns>The weighted addition result.</returns>
        public PradResult VectorWeightedAdd(PradOp opInput1, PradOp opInput2, PradOp opWeights)
        {
            var rows = opInput1.SeedResult.Result.Shape[0];
            var cols = opInput2.SeedResult.Result.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1, angle1) = opInput1.Split(halfCols, axis: 1);
            var (magnitude2, angle2) = opInput2.Split(halfCols, axis: 1);

            var magnitude1Branch = magnitude1.Branch();
            var angle1Branch = angle1.Branch();

            var cosResult = angle1.Cos().Result;
            var sinResult = angle1Branch.Sin().Result;

            var x1 = magnitude1.Mul(cosResult);
            var y1 = magnitude1Branch.Mul(sinResult);

            var magnitude2Branch = magnitude2.Branch();
            var angle2Branch = angle2.Branch();

            var cosResult1 = angle2.Cos().Result;
            var sinResult1 = angle2Branch.Sin().Result;

            var x2 = magnitude2.Mul(cosResult1);
            var y2 = magnitude2Branch.Mul(sinResult1);

            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            var sumXSquared = sumX.PradOp.Square();
            var sumYSquared = sumY.PradOp.Square();
            var magnitudeSquared = sumXSquared.Then(PradOp.AddOp, sumYSquared.Result);
            var resultMagnitude = magnitudeSquared.Then(PradOp.SquareRootOp)
                                                  .Then(PradOp.MulOp, opWeights.SeedResult.Result);
            var resultAngle = sumYBranch.Atan2(sumXBranch.SeedResult.Result);

            var res = resultMagnitude.PradOp.Concat(new[] { resultAngle.Result }, axis: 1);

            return res;
        }

        /// <summary>
        /// Perform a sine-softmax operation.
        /// </summary>
        /// <param name="opInput1">The input.</param>
        /// <returns>The sine-softmax probability distribution output.</returns>
        public PradResult SineSoftmax(PradOp opInput1)
        {
            var sinned = opInput1.Sin();
            var exped = sinned.Then(PradOp.ExpOp);

            // Determine the axis to sum over based on the input dimensions
            int sumAxis = opInput1.CurrentShape.Length - 1;  // Last dimension

            var expedBranch = exped.BranchStack(2);
            var sums = exped.PradOp.Sum(new[] { sumAxis });
            var broadcastedSums = sums.Then(PradOp.BroadcastToOp, opInput1.SeedResult.Result.Shape);
            var denominator = broadcastedSums.PradOp.Add(expedBranch.Pop().BranchInitialTensor);
            var output = denominator.PradOp.DivInto(expedBranch.Pop().BranchInitialTensor);

            return output;
        }

        /// <summary>
        /// Implement a pair-wise sine softmax.
        /// </summary>
        /// <param name="opInput">The input.</param>
        /// <returns>The pairs.</returns>
        /// <exception cref="ArgumentException">Throws if columns are uneven.</exception>
        public PradResult PairwiseSineSoftmax(PradOp opInput)
        {
            var shape = opInput.CurrentShape;
            int numCols = shape[shape.Length - 1];

            if (numCols % 2 != 0)
            {
                throw new ArgumentException("Input tensor must have an even number of columns in its last dimension for pair-wise operation.");
            }

            int m = numCols / 2;

            var firstHalf = opInput.Indexer(":", $":{m}");
            var secondHalf = opInput.Indexer(":", $"{m}:");

            var sinned = opInput.Sin();
            var exped = sinned.PradOp.Exp();

            var expFirst = exped.PradOp.Indexer(":", $":{m}");
            var expSecond = exped.PradOp.Indexer(":", $"{m}:");

            var sumExp = expFirst.PradOp.Add(expSecond.Result);
            var epsilon = new Tensor(sumExp.PradOp.CurrentShape, 1e-9);
            var denominator = sumExp.PradOp.Add(epsilon);

            var outputFirst = expFirst.PradOp.Div(denominator.Result);
            var outputSecond = expSecond.PradOp.Div(denominator.Result);

            return outputFirst.PradOp.Concat(new[] { outputSecond.Result }, axis: -1);
        }

        /// <summary>
        /// A vector scaling operation.
        /// </summary>
        /// <param name="input1">The first input.</param>
        /// <param name="input2">The second input.</param>
        /// <returns>The scaled result.</returns>
        /// <exception cref="ArgumentException">The columns must be even.</exception>
        public PradResult VectorScaling(PradOp input1, PradOp input2)
        {
            var shape = input1.CurrentShape;
            int numCols = shape[shape.Length - 1];

            if (numCols % 2 != 0)
            {
                throw new ArgumentException("Input tensor must have an even number of columns for vector operation.");
            }

            int halfCols = numCols / 2;

            // Split input1 into magnitude and angle
            var magnitudes = input1.Indexer(":", $":{halfCols}");
            var angles = input1.Indexer(":", $"{halfCols}:");

            // Ensure input2 has the correct shape for scaling
            if (!input2.CurrentShape.SequenceEqual(magnitudes.PradOp.CurrentShape))
            {
                throw new ArgumentException("Input2 must have the same shape as the magnitude part of input1.");
            }

            // Scale the magnitudes
            var scaledMagnitudes = magnitudes.PradOp.Mul(input2.SeedResult.Result);

            // Concatenate scaled magnitudes with unchanged angles
            return scaledMagnitudes.PradOp.Concat(new[] { angles.Result }, axis: -1);
        }

        /// <summary>
        /// Implements vector averaging.
        /// </summary>
        /// <param name="input1">The first input.</param>
        /// <param name="input2">The second input.</param>
        /// <returns>The averaged result.</returns>
        /// <exception cref="ArgumentException">Must have even columsn.</exception>
        public PradResult VectorAveraging(PradOp input1, PradOp input2)
        {
            // Ensure inputs have the same shape
            if (!input1.CurrentShape.SequenceEqual(input2.CurrentShape))
            {
                throw new ArgumentException("Input tensors must have the same shape.");
            }

            int cols = input1.CurrentShape[input1.CurrentShape.Length - 1];
            if (cols % 2 != 0)
            {
                throw new ArgumentException("Input tensors must have an even number of columns in the last dimension.");
            }

            int halfCols = cols / 2;

            // Use indexers to access magnitude and angle components
            var magnitude1 = input1.Indexer(":", $":{halfCols}");
            var angle1 = input1.Indexer(":", $"{halfCols}:");
            var magnitude2 = input2.Indexer(":", $":{halfCols}");
            var angle2 = input2.Indexer(":", $"{halfCols}:");

            // Average magnitudes
            var avgMagnitude = magnitude1.PradOp.Add(magnitude2.Result).PradOp.Mul(new Tensor(magnitude1.PradOp.CurrentShape, 0.5));

            // Average angles
            var avgAngle = angle1.PradOp.Add(angle2.Result).PradOp.Mul(new Tensor(angle1.PradOp.CurrentShape, 0.5));

            // Concatenate averaged magnitudes and angles
            return avgMagnitude.PradOp.Concat(new[] { avgAngle.Result }, axis: -1);
        }

        /// <summary>
        /// Implements element-wise inversion.
        /// </summary>
        /// <param name="opInput1">The input.</param>
        /// <returns>The inverted result.</returns>
        public PradResult ElementwiseInversion(PradOp opInput1)
        {
            return opInput1.SubFrom(new Tensor(opInput1.CurrentShape, 1d));
        }

        /// <summary>
        /// Performs a vector attention operation.
        /// </summary>
        /// <param name="vectors">The vectors.</param>
        /// <param name="probabilities">The probabilities.</param>
        /// <returns>The attended to result.</returns>
        /// <exception cref="ArgumentException">Must be compatible.</exception>
        public PradResult VectorAttention(PradOp vectors, PradOp probabilities)
        {
            // Ensure inputs have compatible shapes
            if (vectors.CurrentShape[0] != probabilities.CurrentShape[0] ||
                vectors.CurrentShape[1] / 2 != probabilities.CurrentShape[1])
            {
                throw new ArgumentException("Input shapes are not compatible for vector attention.");
            }

            int cols = vectors.CurrentShape[1];
            int m = cols / 2;

            // Split vectors into magnitude and angle components
            var magnitudes = vectors.Indexer(":", $":{m}");
            var angles = vectors.Indexer(":", $"{m}:");

            // Create constants
            var onePointFive = new Tensor(probabilities.CurrentShape, 1.5f);
            var one = new Tensor(probabilities.CurrentShape, 1.0f);
            var two = new Tensor(probabilities.CurrentShape, 2.0f);
            var halfPi = new Tensor(probabilities.CurrentShape, PradMath.PI / 2);
            var twoPi = new Tensor(probabilities.CurrentShape, 2 * PradMath.PI);

            // Calculate magnitude scaling factor
            var magnitudeScale = probabilities.SubFrom(onePointFive);

            // Scale magnitudes
            var scaledMagnitudes = magnitudes.PradOp.Mul(magnitudeScale.Result);

            // Calculate angle adjustment
            var angleAdjustment = probabilities.SubFrom(one).PradOp.Mul(halfPi).PradOp.Mul(onePointFive);

            // Adjust angles
            var adjustedAngles = angles.PradOp.Add(angleAdjustment.Result).PradOp.Modulus(twoPi);

            // Concatenate scaled magnitudes and adjusted angles
            return scaledMagnitudes.PradOp.Concat(new[] { adjustedAngles.Result }, axis: -1);
        }

        /// <summary>
        /// Performs a leaky ReLU.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <returns>The result of the activation.</returns>
        public PradResult LeakyReLU(PradOp input)
        {
            return input.CustomOperation(
                operation: (inputTensor) =>
                {
                    var result = new Tensor(inputTensor.Shape);
                    for (int i = 0; i < inputTensor.Data.Length; i++)
                    {
                        double x = inputTensor.Data[i];
                        result.Data[i] = x > 0 ? x : 0.01d * x;
                    }

                    return result;
                },
                reverseOperation: (inputTensor, outputTensor, upstreamGradient) =>
                {
                    var gradientTensor = new Tensor(inputTensor.Shape);
                    for (int i = 0; i < inputTensor.Data.Length; i++)
                    {
                        double x = inputTensor.Data[i];
                        double gradient = x > 0 ? 1.0 : 0.01d;
                        gradientTensor.Data[i] = upstreamGradient.Data[i] * gradient;
                    }

                    return new[] { gradientTensor };
                });
        }

        /// <summary>
        /// Performs a cartesian summation operation.
        /// </summary>
        /// <param name="opInput1">The first input.</param>
        /// <param name="opInput2">The second input.</param>
        /// <param name="opWeights">The weights.</param>
        /// <returns>The result of the cartesian summation.</returns>
        public PradResult CartesianSummationOperation(PradOp opInput1, PradOp opInput2, PradOp opWeights)
        {
            var rows = opInput1.SeedResult.Result.Shape[0];
            var cols = opInput1.SeedResult.Result.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1, angle1) = opInput1.Split(halfCols, axis: 1);
            var (magnitude2, angle2) = opInput2.Split(halfCols, axis: 1);

            // Create branches for magnitude1 and magnitude2
            var magnitude1Branch = magnitude1.Branch();
            var magnitude2Branch = magnitude2.Branch();

            var angle1Branch = angle1.Branch();
            var angle2Branch = angle2.Branch();

            // Compute vector components
            var x1 = magnitude1.Mul(angle1.Cos().Result);
            var y1 = magnitude1Branch.Mul(angle1Branch.Sin().Result);
            var x2 = magnitude2.Mul(angle2.Cos().Result);
            var y2 = magnitude2Branch.Mul(angle2Branch.Sin().Result);

            var sumX = x1.Then(PradOp.AddOp, x2.Result);
            var sumY = y1.Then(PradOp.AddOp, y2.Result);

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            var resultMagnitude = sumX.PradOp.Square()
                .Then(PradOp.AddOp, sumY.PradOp.Square().Result)
                .Then(PradOp.SquareRootOp)
                .Then(PradOp.MulOp, opWeights.SeedResult.Result);
            var resultAngle = sumYBranch.Atan2(sumXBranch.SeedResult.Result);

            var resultMagnitudeBranch = resultMagnitude.PradOp.Branch();
            var resultAngleBranch = resultAngle.PradOp.Branch();

            var finalX = resultMagnitude.Then(PradOp.MulOp, resultAngle.PradOp.Cos().Result);
            var finalY = resultMagnitudeBranch.Mul(resultAngleBranch.Sin().Result);

            var sumXTotal = finalX.PradOp.Sum(new[] { 0, 1 }).PradOp.Reshape(1, 1);
            var sumYTotal = finalY.PradOp.Sum(new[] { 0, 1 }).PradOp.Reshape(1, 1);

            var cc = sumXTotal.PradOp.Concat(new[] { sumYTotal.Result }, axis: 1);

            return cc;
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
