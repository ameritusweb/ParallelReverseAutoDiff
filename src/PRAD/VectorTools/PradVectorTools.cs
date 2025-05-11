//------------------------------------------------------------------------------
// <copyright file="PradVectorTools.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.VectorTools
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using ParallelReverseAutoDiff.PRAD;
    using ParallelReverseAutoDiff.PRAD.VectorTools;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// PRAD Tools for Vector Neural Networks.
    /// </summary>
    public class PradVectorTools
    {
        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        public double LearningRate { get; set; } = 0.001d;

        /// <summary>
        /// Generates a vector matrix.
        /// </summary>
        /// <param name="rand">The random generator.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="cols">THe cols.</param>
        /// <param name="target">The target.</param>
        /// <returns>The generated matrix.</returns>
        public Tensor GenerateVectorMatrix(Random rand, int rows, int cols, float target)
        {
            Vector2[][] matrix = new Vector2[rows][];
            for (int i = 0; i < rows; ++i)
            {
                matrix[i] = new Vector2[cols];
                for (int j = 0; j < cols * 2; j += 2)
                {
                    var pair = this.GenerateVectorPair(rand, target);
                    matrix[i][j] = pair.Item1;
                    matrix[i][j + 1] = pair.Item2;
                }
            }

            return matrix.ToInterleavedTensor();
        }

        /// <summary>
        /// Decompose vectors that point to a certain target.
        /// </summary>
        /// <param name="targetX">The target X in Cartesian.</param>
        /// <param name="targetY">The target Y in Cartesian.</param>
        /// <param name="n">The total number of vectors to create.</param>
        /// <returns>The tensor result.</returns>
        public Tensor DecomposeVectors(double targetX, double targetY, int n)
        {
            var decomposer = new VectorDecomposer();

            var vectors = decomposer.DecomposeVector(targetX, targetY, n);

            var arr = decomposer.RandomizeAndFlatten(vectors);

            return new Tensor(new int[] { 1, arr.Length }, arr);
        }

        /// <summary>
        /// Generate a random vector pair toward target.
        /// </summary>
        /// <param name="rand">The random generator.</param>
        /// <param name="target">The target.</param>
        /// <returns>The vector pair.</returns>
        public (Vector2, Vector2) GenerateVectorPair(Random rand, float target)
        {
            var magnitude = (float)rand.NextDouble();
            Vector2 initial = new Vector2(magnitude, (float)rand.NextDouble());
            Vector2 targetVector = new Vector2(magnitude, target);
            var other = targetVector.Sub(initial);
            return (initial, other);
        }

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
        /// Converts from polar to Cartesian.
        /// </summary>
        /// <param name="magnitude">The magnitudes.</param>
        /// <param name="angle">The angles.</param>
        /// <returns>The Cartesian results.</returns>
        public (PradResult, PradResult) PolarToCartesian(PradOp magnitude, PradOp angle)
        {
            var (cosAngle, sinAngle) = angle.DoParallel(
                x => x.Cos(),
                x => x.Sin());

            var (x, y) = magnitude.DoParallel(
                x => x.Mul(cosAngle.Result),
                x => x.Mul(sinAngle.Result));

            return (x, y);
        }

        /// <summary>
        /// Splits an interleaved tensor.
        /// </summary>
        /// <param name="opInput1">The tensor.</param>
        /// <returns>The magnitudes and angles.</returns>
        public (PradResult, PradResult) SplitInterleavedTensor(PradOp opInput1)
        {
            var half_cols = opInput1.CurrentShape[^1] / 2;
            if (opInput1.CurrentShape.Length == 4)
            {
                var (magnitudes1, angles1) = opInput1.DoParallel(
                x => x.Indexer(":", ":", ":", $":{half_cols}"),
                y => y.Indexer(":", ":", ":", $"{half_cols}:"));
                return (magnitudes1, angles1);
            }

            var (magnitudes, angles) = opInput1.DoParallel(
                x => x.Indexer(":", $":{half_cols}"),
                y => y.Indexer(":", $"{half_cols}:"));

            return (magnitudes, angles);
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
        public PradResult VectorMiniDecomposition(PradOp opInput1, PradOp opInput2, PradOp opWeights)
        {
            var num_rows = opInput1.CurrentTensor.Shape[0];
            var num_cols = opInput1.CurrentTensor.Shape[1] / 2;
            var size = num_rows * num_cols;

            var (magnitude, angle) = opInput1.DoParallel(
                x => x.Indexer(":", $":{num_cols}"),
                y => y.Indexer(":", $"{num_cols}:"));
            var magnitudeBranch = magnitude.Branch();
            var angleBranch = angle.Branch();

            var input2_cols = opInput2.CurrentTensor.Shape[1];
            var half_cols = input2_cols / 2;

            var opInput2Branch = opInput2.Branch();

            var opInput2Branches = opInput2.BranchStack(1);

            var w_magnitudes_t = new PradResult[2];
            var w_angles_t = new PradResult[2];
            for (int i = 0; i < 2; i++)
            {
                var branchM = opInput2;

                if (i > 0)
                {
                    branchM = opInput2Branches.Pop();
                }

                var (w_magnitudes_tt, w_angles_tt) = branchM.DoParallel(
                    x => x.Indexer(":", $"{1 + i}:{half_cols}:3"),
                    y => y.Indexer(":", $"{half_cols + 1 + i}::3"));

                w_magnitudes_t[i] = w_magnitudes_tt;
                w_angles_t[i] = w_angles_tt;
            }

            var w_magnitudes_stacked = w_magnitudes_t[0].PradOp.Stack(w_magnitudes_t.Select(f => f.Result).Skip(1).ToArray(), axis: -1);
            var w_angles_stacked = w_angles_t[0].PradOp.Stack(w_angles_t.Select(f => f.Result).Skip(1).ToArray(), axis: -1);

            var w_magnitudes = w_magnitudes_stacked;
            var w_angles = w_angles_stacked;
            var w_magnitudesBranch = w_magnitudes.Branch();
            var w_anglesBranch = w_angles.Branch();

            var w_magnitudesTrans = w_magnitudesBranch.Reshape(1, 2, -1).PradOp.Transpose(new int[] { 0, 2, 1 });
            var w_anglesTrans = w_anglesBranch.Reshape(1, 2, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            var (w_magnitude_pivot_a, w_angle_pivot_a) = opInput2Branch.DoParallel(
                x => x.Indexer(":", $":{half_cols}"),
                y => y.Indexer(":", $"{half_cols}:"));

            var w_magnitude_pivot = w_magnitude_pivot_a.PradOp.Indexer(":", $"::3");
            var w_angle_pivot = w_angle_pivot_a.PradOp.Indexer(":", $"::3");

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

            Tensor addScalar = new Tensor(opWeights.SeedResult.Result.Shape, PradTools.OneHundredth);
            var adjustedWeights = opWeights.Add(addScalar);

            var xPivotAdd = x.Then(PradOp.AddOp, xPivot.Result);
            var yPivotAdd = y.Then(PradOp.AddOp, yPivot.Result);

            var weightsEpsilon = adjustedWeights.Then(PradOp.AddOp, new Tensor(opWeights.SeedResult.Result.Shape, PradTools.Epsilon9));

            var sumX = xPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);
            var sumY = yPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);

            var sumXExpanded = sumX.Then(PradOp.ExpandDimsOp, -1);
            var sumYExpanded = sumY.Then(PradOp.ExpandDimsOp, -1);

            var sumXBranch = sumX.Branch();
            var negativeSumXExpanded = sumXBranch.Mul(new Tensor(sumXBranch.CurrentShape, PradTools.NegativeOne));

            var sumYBranch = sumY.Branch();
            var negativeSumYExpanded = sumYBranch.Mul(new Tensor(sumYBranch.CurrentShape, PradTools.NegativeOne));

            var sumXReshaped = sumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumXReshaped = negativeSumXExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            var x_wReshaped = x_w.Then(PradOp.ReshapeOp, new int[] { 2, size });

            var sumXConcatenated = sumXReshaped.Then(
                PradOp.ConcatOp,
                new[] { negativeSumXReshaped.Result },
                axis: 0);

            var diffX = x_wReshaped.PradOp.SubFrom(sumXConcatenated.Result);

            var sumYReshaped = sumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });
            var negativeSumYReshaped = negativeSumYExpanded.Then(PradOp.ReshapeOp, new int[] { 1, size });

            var y_wReshaped = y_w.Then(PradOp.ReshapeOp, new int[] { 2, size });

            var sumYConcatenated = sumYReshaped.Then(
                PradOp.ConcatOp,
                new[] { negativeSumYReshaped.Result },
                axis: 0);

            var diffY = sumYConcatenated.Then(PradOp.SubOp, y_wReshaped.Result);

            var diffXBranch = diffX.Branch();
            var diffYBranch = diffY.Branch();

            var diffXSquared = diffX.Then(PradOp.SquareOp);
            var diffYSquared = diffY.Then(PradOp.SquareOp);

            var resultMagnitudes = diffXSquared.Then(PradOp.AddOp, diffYSquared.Result)
                                                .Then(PradOp.SquareRootOp);

            var resultAngles = diffYBranch.Atan2(diffXBranch.BranchInitialTensor);

            var resultMagnitudesTrans2 = resultMagnitudes.PradOp.Reshape(1, 2, -1);
            var resultMagnitudesTrans = resultMagnitudesTrans2.PradOp.Transpose(new int[] { 0, 2, 1 });
            var resultAnglesTrans = resultAngles.PradOp.Reshape(1, 2, -1).PradOp.Transpose(new int[] { 0, 2, 1 });

            var reshapedResultMagnitudes = resultMagnitudesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            var reshapedResultAngles = resultAnglesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, -1 });

            var magnitudeReshaped = magnitudeBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var angleReshaped = angleBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var w_magnitude_pivotReshaped = w_magnitude_pivotBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var w_angle_pivotReshaped = w_angle_pivotBranch.Reshape(new int[] { num_rows, num_cols, 1 });
            var w_magnitudesReshaped = w_magnitudesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 2 });
            var w_anglesReshaped = w_anglesTrans.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 2 });
            var resultMagnitudesReshaped = reshapedResultMagnitudes.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 2 });
            var resultAnglesReshaped = reshapedResultAngles.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols, 2 });

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

            var finalOutput = output.Then(PradOp.ReshapeOp, new int[] { num_rows, num_cols * 12 });

            return finalOutput;
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
            var num_rows = opInput1.CurrentTensor.Shape[0];
            var num_cols = opInput1.CurrentTensor.Shape[1] / 2;
            var size = num_rows * num_cols;

            var (magnitude, angle) = opInput1.DoParallel(
                x => x.Indexer(":", $":{num_cols}"),
                y => y.Indexer(":", $"{num_cols}:"));
            var magnitudeBranch = magnitude.Branch();
            var angleBranch = angle.Branch();

            var input2_cols = opInput2.CurrentTensor.Shape[1];
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

            Tensor addScalar = new Tensor(opWeights.SeedResult.Result.Shape, PradTools.OneHundredth);
            var adjustedWeights = opWeights.Add(addScalar);

            var xPivotAdd = x.Then(PradOp.AddOp, xPivot.Result);
            var yPivotAdd = y.Then(PradOp.AddOp, yPivot.Result);

            var weightsEpsilon = adjustedWeights.Then(PradOp.AddOp, new Tensor(opWeights.SeedResult.Result.Shape, PradTools.Epsilon9));

            var sumX = xPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);
            var sumY = yPivotAdd.Then(PradOp.DivOp, weightsEpsilon.Result);

            var sumXExpanded = sumX.Then(PradOp.ExpandDimsOp, -1);
            var sumYExpanded = sumY.Then(PradOp.ExpandDimsOp, -1);

            var sumXBranch = sumX.Branch();
            var negativeSumXExpanded = sumXBranch.Mul(new Tensor(sumXBranch.CurrentShape, PradTools.NegativeOne));

            var sumYBranch = sumY.Branch();
            var negativeSumYExpanded = sumYBranch.Mul(new Tensor(sumYBranch.CurrentShape, PradTools.NegativeOne));

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
        /// <remarks>
        /// input1: [N, M*2]  - each row has M magnitudes then M angles .
        /// input2: [M, P*2]  - each row has P magnitudes then P angles.
        /// weights: [M, P]   - weights for each M->P connection.
        /// </remarks>
        /// <param name="opInput1">The first input.</param>
        /// <param name="opInput2">The second input.</param>
        /// <param name="opWeights">The weights.</param>
        /// <returns>The result of the vector-based matrix multiplication.</returns>
        public PradResult VectorBasedMatrixMultiplication(PradOp opInput1, PradOp opInput2, PradOp opWeights)
        {
            var clonedOpInput1 = opInput1.Branch();
            var clonedOpInput2 = opInput2.Branch();

            var input1 = opInput1.BranchInitialTensor;
            var input2 = opInput2.BranchInitialTensor;

            var rows1 = input1.Shape[0];
            var cols1 = input1.Shape[1];
            var halfCols1 = cols1 / 2;

            var rows2 = input2.Shape[0];
            var cols2 = input2.Shape[1];
            var halfCols2 = cols2 / 2;

            var nN = rows1;
            var mM = halfCols1;
            var pP = halfCols2;

            var magnitudesSeed = opInput1.Indexer(":", $":{mM}");
            var magnitudesOther = opInput2.Indexer(":", $":{pP}");

            var anglesSeed = clonedOpInput1.Indexer(":", $"{mM}:");
            var anglesOther = clonedOpInput2.Indexer(":", $"{pP}:");

            var mBranch1 = magnitudesSeed.Branch();
            var mBranch2 = magnitudesOther.Branch();

            var aBranch1 = anglesSeed.Branch();
            var aBranch2 = anglesOther.Branch();

            var x1 = magnitudesSeed.PradOp.Mul(anglesSeed.PradOp.Cos().Result)
                .PradOp.Reshape(new int[] { nN, mM, 1 }).PradOp.Tile(new int[] { 1, 1, pP });
            var x2 = magnitudesOther.PradOp.Mul(anglesOther.PradOp.Cos().Result)
                .PradOp.Reshape(new int[] { 1, mM, pP }).PradOp.Tile(new int[] { nN, 1, 1 });
            var y1 = mBranch1.Mul(aBranch1.Sin().Result)
                .PradOp.Reshape(new int[] { nN, mM, 1 }).PradOp.Tile(new int[] { 1, 1, pP });
            var y2 = mBranch2.Mul(aBranch2.Sin().Result)
                .PradOp.Reshape(new int[] { 1, mM, pP }).PradOp.Tile(new int[] { nN, 1, 1 });

            var weightsTiled = opWeights.Reshape(new[] { 1, mM, pP })
                                    .PradOp.Tile(new[] { nN, 1, 1 });

            var wtBranch = weightsTiled.Branch();

            var negativeWeights = weightsTiled.PradOp.LessThan(new Tensor(new[] { nN, mM, pP }, PradTools.Zero));

            var negativeWeightsBranch = negativeWeights.Branch();

            var x1Branch = x1.Branch();

            var x2Branch = x2.Branch();

            var subx2x1 = x1Branch.SubFrom(x2.Result);

            var y1Branch = y1.Branch();

            var y2Branch = y2.Branch();

            var suby2y1 = y1Branch.SubFrom(y2.Result);

            var deltaX = x1.PradOp.Sub(x2Branch.CurrentTensor).PradOp.Where(negativeWeights.Result, subx2x1.Result);
            var deltaY = y1.PradOp.Sub(y2Branch.CurrentTensor).PradOp.Where(negativeWeightsBranch.CurrentTensor, suby2y1.Result);

            var deltaXBranch = deltaX.Branch();
            var deltaYBranch = deltaY.Branch();

            var diffMagnitudes = deltaX.PradOp.Square()
                .PradOp.Add(deltaY.PradOp.Square().Result)
                .PradOp.SquareRoot()
                .PradOp.Div(new Tensor(new int[] { nN, mM, pP }, mM)); // [N, M, P]
            var diffAngles = deltaYBranch.Atan2(deltaXBranch.CurrentTensor);  // [N, M, P]

            var daBranch = diffAngles.Branch();

            var abs = wtBranch.Abs();

            var weightedMagnitudes = diffMagnitudes.PradOp.Mul(abs.Result);  // [N, M, P]
            var wmBranch = weightedMagnitudes.Branch();

            var weightedX = weightedMagnitudes.PradOp.Mul(diffAngles.PradOp.Cos().Result);  // [N, M, P]
            var weightedY = wmBranch.Mul(daBranch.Sin().Result);  // [N, M, P]

            var sumX = weightedX.PradOp.Sum(new int[] { 1 });  // [N, P]
            var sumY = weightedY.PradOp.Sum(new int[] { 1 });  // [N, P]

            var sumXBranch = sumX.Branch();
            var sumYBranch = sumY.Branch();

            var finalMagnitudes = sumX.PradOp.Square()
                .PradOp.Add(sumY.PradOp.Square().Result)
                .PradOp.SquareRoot();  // [N, P]
            var finalAngles = sumYBranch.Atan2(sumXBranch.CurrentTensor);  // [N, P]

            var finalResult = finalMagnitudes.PradOp.Concat(new[] { finalAngles.Result }, 1);  // [N, P*2]

            return finalResult;
        }

        /// <summary>
        /// Calculates contribution probabilities for vectors based on their influence on the decision boundary at π/2.
        /// </summary>
        /// <param name="input">Input tensor with magnitudes in left half and angles in right half.</param>
        /// <param name="temperature">The temperature.</param>
        /// <returns>Tensor containing probability of contribution for each vector.</returns>
        public PradResult CalculateVectorContributionProbabilities(PradOp input, double temperature = 1.0d)
        {
            var half_cols = input.CurrentShape[^1] / 2;

            // Split into magnitudes and angles
            var (magnitudes, angles) = input.DoParallel(
                x => x.Indexer(":", $":{half_cols}"),
                y => y.Indexer(":", $"{half_cols}:"));

            // Calculate total resultant vector components
            var (cosAngles, sinAngles) = angles.PradOp.DoParallel(
                x => x.Cos(),
                y => y.Sin());

            var (xComponents, yComponents) = magnitudes.PradOp.DoParallel(
                x => x.Mul(cosAngles.Result),
                y => y.Mul(sinAngles.Result));

            // Calculate resultant magnitude
            var xComponentsSquared = xComponents.PradOp.Square();
            var yComponentsSquared = yComponents.PradOp.Square();
            var resultantMagnitudeSquared = xComponentsSquared.Then(PradOp.AddOp, yComponentsSquared.Result);
            var resultantMagnitude = resultantMagnitudeSquared.Then(PradOp.SquareRootOp);

            // Calculate angle deviations from π/2
            var piOverTwo = new Tensor(angles.PradOp.CurrentShape, (float)(Math.PI / 2));
            var angleDeviations = angles.PradOp.Sub(piOverTwo);
            var absoluteDeviations = angleDeviations.Then(PradOp.AbsOp);

            // Normalize deviations to range 0-1
            var normalizedDeviations = absoluteDeviations.Then(
                PradOp.DivOp,
                new Tensor(absoluteDeviations.PradOp.CurrentShape, (float)(Math.PI / 2)));

            // Calculate relative magnitudes
            var relativeMagnitudes = magnitudes.PradOp.Div(resultantMagnitude.Result);

            // Combine magnitude and angle effects
            var contributions = relativeMagnitudes.Then(PradOp.MulOp, normalizedDeviations.Result);

            // Apply sine-softmax to get probabilities
            return this.SineSoftmax(contributions.PradOp, temperature);
        }

        /// <summary>
        /// Performs a vector-based transpose operation.
        /// </summary>
        /// <param name="input">The tensor to transpose.</param>
        /// <returns>The result.</returns>
        public PradResult VectorBasedTranspose(PradOp input)
        {
            var rows = input.CurrentTensor.Shape[0];
            var cols = input.CurrentTensor.Shape[1];
            var halfCols = cols / 2;

            var branch = input.Branch();

            // Split magnitudes and angles
            var magnitudes = input.Indexer(":", $":{halfCols}");  // [rows, halfCols]
            var angles = branch.Indexer(":", $"{halfCols}:");      // [rows, halfCols]

            // Transpose each half separately
            var transposedMagnitudes = magnitudes.PradOp.Transpose(new[] { 1, 0 });  // [halfCols, rows]
            var transposedAngles = angles.PradOp.Transpose(new[] { 1, 0 });          // [halfCols, rows]

            // Concatenate along axis 1 to get final result
            return transposedMagnitudes.PradOp.Concat(new[] { transposedAngles.Result }, axis: 1);
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
            var rows = opInput1.CurrentTensor.Shape[0];
            var cols = opInput2.CurrentTensor.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1R, angle1R) = this.SplitInterleavedTensor(opInput1);
            var (magnitude2R, angle2R) = this.SplitInterleavedTensor(opInput2);

            var magnitude1 = magnitude1R.PradOp;
            var angle1 = angle1R.PradOp;
            var magnitude2 = magnitude2R.PradOp;
            var angle2 = angle2R.PradOp;

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
        /// Perform a vector add operation.
        /// </summary>
        /// <param name="opInput1">The first input.</param>
        /// <param name="opInput2">The second input.</param>
        /// <returns>The addition result.</returns>
        public PradResult VectorAdd(PradOp opInput1, PradOp opInput2)
        {
            var rows = opInput1.CurrentTensor.Shape[0];
            var cols = opInput2.CurrentTensor.Shape[1];
            var halfCols = cols / 2;

            var (magnitude1R, angle1R) = this.SplitInterleavedTensor(opInput1);
            var (magnitude2R, angle2R) = this.SplitInterleavedTensor(opInput2);

            var magnitude1 = magnitude1R.PradOp;
            var angle1 = angle1R.PradOp;
            var magnitude2 = magnitude2R.PradOp;
            var angle2 = angle2R.PradOp;

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
            var resultMagnitude = magnitudeSquared.Then(PradOp.SquareRootOp);
            var resultAngle = sumYBranch.Atan2(sumXBranch.SeedResult.Result);

            var res = resultMagnitude.PradOp.Concat(new[] { resultAngle.Result }, axis: 1);

            return res;
        }

        /// <summary>
        /// Performs a matrix multiplication between two tensors.
        /// </summary>
        /// <param name="opInput1">The first input.</param>
        /// <param name="opInput2">The second input.</param>
        /// <returns>The result of the matrix multiplication.</returns>
        public PradResult MatrixMultiplication(PradOp opInput1, PradOp opInput2)
        {
            return opInput1.MatMul(opInput2.CurrentTensor);
        }

        /// <summary>
        /// Performs a broadcasting add operation.
        /// </summary>
        /// <param name="opInput1">The first tensor.</param>
        /// <param name="toBroadcast">The tensor to broadcast.</param>
        /// <returns>The result of add broadcasting.</returns>
        public PradResult AddBroadcasting(PradOp opInput1, PradOp toBroadcast)
        {
            var broadcasted = toBroadcast.BroadcastTo(opInput1.CurrentShape);
            return opInput1.Add(broadcasted.Result);
        }

        /// <summary>
        /// Perform a sine-softmax operation.
        /// </summary>
        /// <param name="opInput1">The input.</param>
        /// <param name="temperature">The temperature.</param>
        /// <returns>The sine-softmax probability distribution output.</returns>
        public PradResult SineSoftmax(PradOp opInput1, double temperature = 1.0d)
        {
            var shape = opInput1.CurrentTensor.Shape;

            var scaled = opInput1.Mul(new Tensor(opInput1.CurrentShape, PradTools.One / PradTools.Cast(temperature)));

            var sinned = scaled.PradOp.Sin();
            var exped = sinned.Then(PradOp.ExpOp);

            // Determine the axis to sum over based on the input dimensions
            int sumAxis = opInput1.CurrentShape.Length - 1;  // Last dimension

            var expedBranch = exped.BranchStack(2);
            var sums = exped.PradOp.Sum(new[] { sumAxis });
            var broadcastedSums = sums.Then(PradOp.BroadcastToOp, shape);
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

            var opInputBranch = opInput.Branch();
            var firstHalf = opInput.Indexer(":", $":{m}");
            var secondHalf = opInputBranch.Indexer(":", $"{m}:");

            var sinFirst = firstHalf.PradOp.Sin();
            var sinSecond = secondHalf.PradOp.Sin();

            var sinFirstExp = sinFirst.PradOp.Exp();
            var sinSecondExp = sinSecond.PradOp.Exp();

            var sinFirstBranchExp = sinFirstExp.Branch();
            var sinSecondBranchExp = sinSecondExp.Branch();

            var sumExp = sinFirstExp.PradOp.Add(sinSecondExp.Result);

            var epsilon = new Tensor(sumExp.PradOp.CurrentShape, PradTools.Epsilon9);
            var denominator = sumExp.PradOp.Add(epsilon);

            var denominatorBranch = denominator.Branch();
            var outputFirst = denominator.PradOp.DivInto(sinFirstBranchExp.BranchInitialTensor);
            var outputSecond = denominatorBranch.DivInto(sinSecondBranchExp.BranchInitialTensor);

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
            var (magnitude1, angle1) = this.SplitInterleavedTensor(input1);
            var (magnitude2, angle2) = this.SplitInterleavedTensor(input2);

            // Average magnitudes
            var magnitudeSum = magnitude1.PradOp.Add(magnitude2.Result);
            var avgMagnitude = magnitudeSum.PradOp.Mul(new Tensor(magnitude1.PradOp.CurrentShape, PradTools.Half));

            // Average angles
            var angleSum = angle1.PradOp.Add(angle2.Result);
            var avgAngle = angleSum.PradOp.Mul(new Tensor(angle1.PradOp.CurrentShape, PradTools.Half));

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
            return opInput1.SubFrom(new Tensor(opInput1.CurrentShape, PradTools.One));
        }

        /// <summary>
        /// Performs a vector attention operation.
        /// </summary>
        /// <param name="vectors">The vectors.</param>
        /// <param name="probabilitiesBoth">The probabilities.</param>
        /// <returns>The attended to result.</returns>
        /// <exception cref="ArgumentException">Must be compatible.</exception>
        public PradResult VectorAttention(PradOp vectors, PradOp probabilitiesBoth)
        {
            var halfCols = probabilitiesBoth.CurrentShape[^1] / 2;
            var probabilities = probabilitiesBoth.Indexer("...", $"{halfCols}:").PradOp;
            var probabilitiesBranch = probabilities.Branch();

            // Ensure inputs have compatible shapes
            if (vectors.CurrentShape[0] != probabilities.CurrentShape[0] ||
                vectors.CurrentShape[1] / 2 != probabilities.CurrentShape[1])
            {
                throw new ArgumentException("Input shapes are not compatible for vector attention.");
            }

            int cols = vectors.CurrentShape[1];
            int m = cols / 2;

            // Split vectors into magnitude and angle components
            var (magnitudes, angles) = this.SplitInterleavedTensor(vectors);

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
            var angleAdjustment = probabilitiesBranch.SubFrom(one).PradOp.Mul(halfPi).PradOp.Mul(onePointFive);

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
                        var x = inputTensor.Data[i];
                        result.Data[i] = x > 0 ? x : PradTools.OneHundredth * x;
                    }

                    return result;
                },
                reverseOperation: (inputTensor, outputTensor, upstreamGradient) =>
                {
                    var gradientTensor = new Tensor(inputTensor.Shape);
                    for (int i = 0; i < inputTensor.Data.Length; i++)
                    {
                        var x = inputTensor.Data[i];
                        var gradient = x > 0 ? PradTools.One : PradTools.OneHundredth;
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

        /// <summary>
        /// Perform a custom 2-D vector convolution.
        /// The magnitude dimensions should be [batch_size, rows, columns, 1]
        /// The angle dimensions should be [batch_size, rows, columns, 1]
        /// The filterMagnitude dimensions should be [1, 1, 1, 4]
        /// The filterAngle dimensions should be [1, 1, 1, 4].
        /// </summary>
        /// <param name="magnitude">The magnitudes.</param>
        /// <param name="angle">The angles.</param>
        /// <param name="filterMagnitude">The filter magnitudes.</param>
        /// <param name="filterAngle">The angle magnitudes.</param>
        /// <returns>The result of the convolution.</returns>
        public PradResult CustomVectorConvolution(PradOp magnitude, PradOp angle, PradOp filterMagnitude, PradOp filterAngle)
        {
            var (x, y) = this.PolarToCartesian(magnitude, angle);
            var (filterX, filterY) = this.PolarToCartesian(filterMagnitude, filterAngle);

            var xPatches = x.PradOp.ExtractPatches(new int[] { 2, 2 }, new int[] { 1, 1 }, "SAME");
            var yPatches = y.PradOp.ExtractPatches(new int[] { 2, 2 }, new int[] { 1, 1 }, "SAME");

            var filterXR = filterX.PradOp.Reshape(new int[] { 1, 1, 1, 4 });
            var filterYR = filterY.PradOp.Reshape(new int[] { 1, 1, 1, 4 });

            var tileMultiples = new int[] { xPatches.PradOp.CurrentShape[0], xPatches.PradOp.CurrentShape[1], 1, 1 };
            var tiledFilterX = filterXR.PradOp.Tile(tileMultiples);
            var tiledFilterY = filterYR.PradOp.Tile(tileMultiples);

            var tileMultiples2 = new int[] { 1, 1, xPatches.PradOp.CurrentShape[2], 1 };
            var tiledFilterX1 = tiledFilterX.PradOp.Tile(tileMultiples2);
            var tiledFilterY1 = tiledFilterY.PradOp.Tile(tileMultiples2);

            var dotProductX = xPatches.PradOp.Mul(tiledFilterX1.Result);
            var dotProductY = yPatches.PradOp.Mul(tiledFilterY1.Result);
            var dotProduct = dotProductX.PradOp.Add(dotProductY.Result);

            var dotProductR = dotProduct.PradOp.Reshape(new int[] { dotProduct.PradOp.CurrentShape[1] * dotProduct.PradOp.CurrentShape[2], dotProduct.PradOp.CurrentShape[3] });

            var summedResult = dotProductR.PradOp.Sum(new int[] { 0 });

            return summedResult;
        }

        /// <summary>
        /// Calculate the squared arc length Euclidean loss.
        /// </summary>
        /// <param name="predictions">The Cartesian prediction.</param>
        /// <param name="targetAngle">The target angle.</param>
        /// <returns>The result of the loss function.</returns>
        public PradResult SquaredArclengthEuclideanLoss(PradOp predictions, double targetAngle)
        {
            return predictions.CustomOperation(
                operation: (inputTensor) =>
                {
                    var output = new Tensor(new[] { 1, 1 });
                    double xOutput = inputTensor[0, 0];
                    double yOutput = inputTensor[0, 1];

                    double magnitude = Math.Sqrt((xOutput * xOutput) + (yOutput * yOutput));
                    double actualAngle = Math.Atan2(yOutput, xOutput);

                    double xTarget = Math.Cos(targetAngle) * magnitude;
                    double yTarget = Math.Sin(targetAngle) * magnitude;

                    double xTargetUnnormalized = Math.Cos(targetAngle);
                    double yTargetUnnormalized = Math.Sin(targetAngle);

                    double radius = magnitude;
                    double dotProduct = (xOutput * xTarget) + (yOutput * yTarget);

                    double normalizedDotProduct = dotProduct / (radius * radius);
                    normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

                    double theta = Math.Acos(normalizedDotProduct);

                    double distanceXQuad = (0.75d * Math.Pow(xOutput, 2)) - (1.5d * xOutput * xTargetUnnormalized);
                    double distanceYQuad = (0.75d * Math.Pow(yOutput, 2)) - (1.5d * yOutput * yTargetUnnormalized);
                    double distanceAccum = distanceXQuad + distanceYQuad;

                    double arcLength = Math.Pow(radius * theta, 2);

                    double lossMagnitude = (arcLength + distanceAccum) / 2d;

                    output[0, 0] = (float)lossMagnitude;

                    return output;
                },
                reverseOperation: (inputTensor, outputTensor, upstreamGradient) =>
                {
                    var dPredictions = new Tensor(new[] { 1, 2 });
                    double xOutput = inputTensor[0, 0];
                    double yOutput = inputTensor[0, 1];

                    double magnitude = Math.Sqrt((xOutput * xOutput) + (yOutput * yOutput));
                    double actualAngle = Math.Atan2(yOutput, xOutput);

                    double xTarget = Math.Cos(targetAngle) * magnitude;
                    double yTarget = Math.Sin(targetAngle) * magnitude;

                    double xTargetUnnormalized = Math.Cos(targetAngle);
                    double yTargetUnnormalized = Math.Sin(targetAngle);

                    double radius = magnitude;
                    double dotProduct = (xOutput * xTarget) + (yOutput * yTarget);

                    double normalizedDotProduct = dotProduct / (radius * radius);
                    normalizedDotProduct = Math.Clamp(normalizedDotProduct, -1.0, 1.0);

                    double theta = Math.Acos(normalizedDotProduct);
                    double denominator = Math.Sqrt(1 - (normalizedDotProduct * normalizedDotProduct));

                    double gradXOutput = xTarget * theta / denominator;
                    double gradYOutput = yTarget * theta / denominator;

                    double dLoss_dX = (xOutput - xTargetUnnormalized) * (3d / 2d);
                    double dLoss_dY = (yOutput - yTargetUnnormalized) * (3d / 2d);

                    var anglesTensor = new Tensor(new[] { 1, 2 }, new[] { PradTools.Cast(actualAngle), PradTools.Cast(targetAngle) });
                    (double cX, double cY) = anglesTensor.CalculateCoefficient();
                    dPredictions[0, 0] = PradTools.Cast(cX * Math.Abs(gradXOutput + dLoss_dX) * upstreamGradient[0, 0]);
                    dPredictions[0, 1] = PradTools.Cast(cY * Math.Abs(gradYOutput + dLoss_dY) * upstreamGradient[0, 0]);

                    return new[] { dPredictions };
                });
        }

        /// <summary>
        /// Computes an ordering loss based on direction similarity.
        /// </summary>
        /// <param name="tensorOp">The tensor with reference vectors at indices 0 and ^1.</param>
        /// <param name="margin">The margin.</param>
        /// <returns>A result.</returns>
        public PradResult ComputeOrderingLoss(PradOp tensorOp, float margin = 0.1f)
        {
            // Split tensor into magnitudes and angles
            var halfLength = tensorOp.CurrentShape[1] / 2;
            var n = tensorOp.CurrentTensor.Shape[0];
            var nMinusOne = n - 1;

            // Get magnitudes and angles using indexer
            var tensorOpBranch = tensorOp.Branch();
            var magnitudes = tensorOpBranch.Indexer(":", $"0:{halfLength}");
            var angles = tensorOp.Indexer(":", $"{halfLength}:");

            // Convert polar to cartesian coordinates
            var anglesBranch = angles.Branch();
            var cosAngles = anglesBranch.Cos();
            var sinAngles = angles.PradOp.Sin();

            // Multiply by magnitudes
            var xCoords = cosAngles.Then(result => result.PradOp.Mul(magnitudes.Result));
            var yCoords = sinAngles.Then(result => result.PradOp.Mul(magnitudes.Result));

            var xCoordsBranch = xCoords.BranchStack(2);
            var yCoordsBranch = yCoords.BranchStack(2);

            // Get reference tensors (first and last)
            var refA_x = xCoords.PradOp.Indexer("0:1", ":");
            var refA_y = yCoordsBranch.Pop().Indexer("0:1", ":");
            var refZ_x = xCoordsBranch.Pop().Indexer($"{nMinusOne}:{n}", ":");
            var refZ_y = yCoordsBranch.Pop().Indexer($"{nMinusOne}:{n}", ":");

            // Calculate angles relative to x-axis for reference points
            var refAAngle = refA_y.PradOp.Atan2(refA_x.Result);
            var refZAngle = refZ_y.PradOp.Atan2(refZ_x.Result);

            var refAAngleBranch = refAAngle.Branch();
            var refZAngleBranch = refZAngle.Branch();

            var refAAngleB = refAAngle.PradOp.BroadcastTo(yCoords.PradOp.CurrentShape);
            var refZAngleB = refZAngle.PradOp.BroadcastTo(yCoords.PradOp.CurrentShape);

            // Calculate angles for all points
            var allAngles = yCoords.Then(result => result.PradOp.Atan2(xCoordsBranch.Pop().CurrentTensor));

            // Branch for multiple uses
            var allAnglesBranch = allAngles.Branch();

            // Calculate angle differences from refA
            var angleFromA = allAngles.PradOp
                .Sub(refAAngleB.Result)
                .Then(PradOp.ModulusOp, new Tensor(allAngles.PradOp.CurrentShape, PradTools.Cast(2 * Math.PI)));

            var angleFromABranch = angleFromA.Branch();

            // Calculate angle differences from refZ
            var angleFromZ = allAnglesBranch
                .Sub(refZAngleB.Result)
                .Then(PradOp.ModulusOp, new Tensor(allAngles.PradOp.CurrentShape, PradTools.Cast(2 * Math.PI)));

            // Calculate total angle span
            var totalSpan = refZAngleBranch.Sub(refAAngleBranch.CurrentTensor)
                .Then(PradOp.ModulusOp, new Tensor(refZAngleBranch.CurrentShape, PradTools.Cast(2 * Math.PI)));

            var totalSpanBranch = totalSpan.Branch();

            // Expected spacing between consecutive tensors
            var n1 = new Tensor(totalSpan.PradOp.CurrentShape, tensorOp.CurrentShape[0] - 1);
            var expectedSpacing = totalSpan.Then(result => result.PradOp.Div(n1));

            // Generate pairs for comparison
            var indices = Enumerable.Range(0, tensorOp.CurrentShape[0]).ToArray();
            var pairs = this.GeneratePairs(indices);

            PradResult totalLossResult = null!;

            var pairLength = pairs.Length;

            var expectedSpacingBranches = expectedSpacing.BranchStack(pairLength - 1);

            var branchCount = (pairLength * 2) - 1;
            var allAnglesBranches = allAngles.BranchStack(branchCount);

            int ii = 0;
            foreach (var (i, j) in pairs)
            {
                ii++;

                // Get angles for pair
                var angleI = ii == 1 ? allAngles.PradOp.Indexer($"{i}:{i + 1}", ":") : allAnglesBranches.Pop().Indexer($"{i}:{i + 1}", ":");
                var angleJ = allAnglesBranches.Pop().Indexer($"{j}:{j + 1}", ":");

                var shape = angleJ.PradOp.CurrentShape;

                // Calculate actual spacing
                var actualSpacing = angleI.PradOp
                    .SubFrom(angleJ.Result)
                    .Then(PradOp.ModulusOp, new Tensor(shape, PradTools.Cast(2 * Math.PI)));

                // Expected spacing based on indices
                var expectedTotal = ii == 1 ? expectedSpacing.Then(result =>
                    result.PradOp.Mul(new Tensor(shape, j - i))) :
                    expectedSpacingBranches.Pop().Mul(new Tensor(shape, j - i));

                // Calculate loss component
                var spacingDiff = actualSpacing.Then(result =>
                    result.PradOp.Sub(expectedTotal.Result).Then(PradOp.AbsOp));

                var marginTensor = new Tensor(shape, margin);
                var lossTerm = spacingDiff.Then(result =>
                    result.PradOp.Sub(marginTensor).Then(PradOp.MaxOp, new Tensor(shape, PradTools.Zero))).PradOp.Square();

                totalLossResult = ii == 1 ? lossTerm : totalLossResult.PradOp.Add(lossTerm.Result);
            }

            var totalSpanBranchB = totalSpanBranch.BroadcastTo(angleFromABranch.CurrentShape);

            // Normalize angles relative to both references
            var normalizedFromA = angleFromABranch.Div(totalSpanBranchB.PradOp.CurrentTensor);

            var normalizedFromZ = angleFromZ.Then(result =>
                result.PradOp.Div(totalSpanBranchB.PradOp.CurrentTensor));

            // Expected positions (0 = closest to A, 1 = closest to Z)
            var indicesT = new Tensor(
                angleFromABranch.CurrentShape,
                Enumerable.Range(0, n).SelectMany(i =>
                    Enumerable.Repeat(PradTools.Cast(i) / (n - 1), halfLength)).ToArray());

            // Penalize deviations from expected positions relative to both references
            var fromADiff = normalizedFromA.Then(result =>
                result.PradOp.Sub(indicesT).Then(PradOp.AbsOp)).PradOp.Square();

            var fromZDiff = normalizedFromZ.Then(result =>
                result.PradOp.Add(indicesT).PradOp.Sub(new Tensor(indicesT.Shape, PradTools.One)).Then(PradOp.AbsOp)).PradOp.Square();

            // Combine both penalties
            var globalSpacingDiff = fromADiff.Then(result =>
                result.PradOp.Add(fromZDiff.Result))
                .PradOp.Mean(0)
                .PradOp.BroadcastTo(new int[] { 1, halfLength });

            // Add to total loss
            totalLossResult = totalLossResult.PradOp.Add(globalSpacingDiff.Result);

            var numPairs = new Tensor(totalLossResult.PradOp.CurrentShape, pairs.Length);
            var mseLoss = totalLossResult.Then(result => result.PradOp.Div(numPairs));

            return mseLoss;
        }

        /// <summary>
        /// Computes an ordering loss based on direction similarity based on a sum per row.
        /// </summary>
        /// <param name="tensorOp">The tensor with reference vectors at indices 0 and ^1.</param>
        /// <param name="margin">The margin.</param>
        /// <returns>A result.</returns>
        public PradResult ComputeOrderingLoss2(PradOp tensorOp, float margin = 0.1f)
        {
            // Split tensor into magnitudes and angles
            var halfLength = tensorOp.CurrentShape[1] / 2;
            var n = tensorOp.CurrentTensor.Shape[0];
            var nMinusOne = n - 1;

            // Get magnitudes and angles using indexer
            var tensorOpBranch = tensorOp.Branch();
            var magnitudes = tensorOpBranch.Indexer(":", $"0:{halfLength}");
            var angles = tensorOp.Indexer(":", $"{halfLength}:");

            // Convert polar to cartesian coordinates
            var anglesBranch = angles.Branch();
            var cosAngles = anglesBranch.Cos();
            var sinAngles = angles.PradOp.Sin();

            // Multiply by magnitudes
            var xCoords = cosAngles.Then(result => result.PradOp.Mul(magnitudes.Result))
                .PradOp.SumRows();
            var yCoords = sinAngles.Then(result => result.PradOp.Mul(magnitudes.Result))
                .PradOp.SumRows();

            var xCoordsBranch = xCoords.BranchStack(2);
            var yCoordsBranch = yCoords.BranchStack(2);

            // Get reference tensors (first and last)
            var refA_x = xCoords.PradOp.Indexer("0:1", ":");
            var refA_y = yCoordsBranch.Pop().Indexer("0:1", ":");
            var refZ_x = xCoordsBranch.Pop().Indexer($"{nMinusOne}:{n}", ":");
            var refZ_y = yCoordsBranch.Pop().Indexer($"{nMinusOne}:{n}", ":");

            // Calculate angles relative to x-axis for reference points
            var refAAngle = refA_y.PradOp.Atan2(refA_x.Result);
            var refZAngle = refZ_y.PradOp.Atan2(refZ_x.Result);

            var refAAngleBranch = refAAngle.Branch();
            var refZAngleBranch = refZAngle.Branch();

            var refAAngleB = refAAngle.PradOp.BroadcastTo(yCoords.PradOp.CurrentShape);
            var refZAngleB = refZAngle.PradOp.BroadcastTo(yCoords.PradOp.CurrentShape);

            // Calculate angles for all points
            var allAngles = yCoords.Then(result => result.PradOp.Atan2(xCoordsBranch.Pop().CurrentTensor));

            // Branch for multiple uses
            var allAnglesBranch = allAngles.Branch();

            // Calculate angle differences from refA
            var angleFromA = allAngles.PradOp
                .Sub(refAAngleB.Result)
                .Then(PradOp.ModulusOp, new Tensor(allAngles.PradOp.CurrentShape, PradTools.Cast(2 * Math.PI)));

            var angleFromABranch = angleFromA.Branch();

            // Calculate angle differences from refZ
            var angleFromZ = allAnglesBranch
                .Sub(refZAngleB.Result)
                .Then(PradOp.ModulusOp, new Tensor(allAngles.PradOp.CurrentShape, PradTools.Cast(2 * Math.PI)));

            // Calculate total angle span
            var totalSpan = refZAngleBranch.Sub(refAAngleBranch.CurrentTensor)
                .Then(PradOp.ModulusOp, new Tensor(refZAngleBranch.CurrentShape, PradTools.Cast(2 * Math.PI)));

            var totalSpanBranch = totalSpan.Branch();

            // Expected spacing between consecutive tensors
            var n1 = new Tensor(totalSpan.PradOp.CurrentShape, tensorOp.CurrentShape[0] - 1);
            var expectedSpacing = totalSpan.Then(result => result.PradOp.Div(n1));

            // Generate pairs for comparison
            var indices = Enumerable.Range(0, tensorOp.CurrentShape[0]).ToArray();
            var pairs = this.GeneratePairs(indices);

            PradResult totalLossResult = null!;

            var pairLength = pairs.Length;

            var expectedSpacingBranches = expectedSpacing.BranchStack(pairLength - 1);

            var branchCount = (pairLength * 2) - 1;
            var allAnglesBranches = allAngles.BranchStack(branchCount);

            int ii = 0;
            foreach (var (i, j) in pairs)
            {
                ii++;

                // Get angles for pair
                var angleI = ii == 1 ? allAngles.PradOp.Indexer($"{i}:{i + 1}", ":") : allAnglesBranches.Pop().Indexer($"{i}:{i + 1}", ":");
                var angleJ = allAnglesBranches.Pop().Indexer($"{j}:{j + 1}", ":");

                var shape = angleJ.PradOp.CurrentShape;

                // Calculate actual spacing
                var actualSpacing = angleI.PradOp
                    .SubFrom(angleJ.Result)
                    .Then(PradOp.ModulusOp, new Tensor(shape, PradTools.Cast(2 * Math.PI)));

                // Expected spacing based on indices
                var expectedTotal = ii == 1 ? expectedSpacing.Then(result =>
                    result.PradOp.Mul(new Tensor(shape, j - i))) :
                    expectedSpacingBranches.Pop().Mul(new Tensor(shape, j - i));

                // Calculate loss component
                var spacingDiff = actualSpacing.Then(result =>
                    result.PradOp.Sub(expectedTotal.Result).Then(PradOp.AbsOp));

                var marginTensor = new Tensor(shape, margin);
                var lossTerm = spacingDiff.Then(result =>
                    result.PradOp.Sub(marginTensor).Then(PradOp.MaxOp, new Tensor(shape, PradTools.Zero))).PradOp.Square();

                totalLossResult = ii == 1 ? lossTerm : totalLossResult.PradOp.Add(lossTerm.Result);
            }

            var totalSpanBranchB = totalSpanBranch.BroadcastTo(angleFromABranch.CurrentShape);

            // Normalize angles relative to both references
            var normalizedFromA = angleFromABranch.Div(totalSpanBranchB.PradOp.CurrentTensor);

            var normalizedFromZ = angleFromZ.Then(result =>
                result.PradOp.Div(totalSpanBranchB.PradOp.CurrentTensor));

            // Expected positions (0 = closest to A, 1 = closest to Z)
            var indicesT = new Tensor(
                angleFromABranch.CurrentShape,
                Enumerable.Range(0, n).Select(i => PradTools.Cast(i) / (n - 1)).ToArray());

            // Penalize deviations from expected positions relative to both references
            var fromADiff = normalizedFromA.Then(result =>
                result.PradOp.Sub(indicesT).Then(PradOp.AbsOp)).PradOp.Square();

            var fromZDiff = normalizedFromZ.Then(result =>
                result.PradOp.Add(indicesT).PradOp.Sub(new Tensor(indicesT.Shape, PradTools.One)).Then(PradOp.AbsOp)).PradOp.Square();

            // Combine both penalties
            var globalSpacingDiff = fromADiff.Then(result =>
                result.PradOp.Add(fromZDiff.Result))
                .PradOp.Mean(0)
                .PradOp.BroadcastTo(new int[] { 1, 1 });

            // Add to total loss
            totalLossResult = totalLossResult.PradOp.Add(globalSpacingDiff.Result);

            var numPairs = new Tensor(totalLossResult.PradOp.CurrentShape, pairs.Length);
            var mseLoss = totalLossResult.Then(result => result.PradOp.Div(numPairs));

            return mseLoss;
        }

        /// <summary>
        /// Use gradient descent to adjust tensors.
        /// </summary>
        /// <param name="tensorOps">The tensors to adjust.</param>
        /// <returns>The list of prad op.</returns>
        public List<PradOp> GradientDescent(params PradOp[] tensorOps)
        {
            List<PradOp> newOps = new List<PradOp>();
            var clipper = PradClipper.CreateGradientClipper();
            foreach (var tensor in tensorOps)
            {
                var clippedGradient = clipper.ClipGradients(tensor.SeedGradient);
                PradOp newOp = new PradOp(tensor.BranchInitialTensor.ElementwiseSub(clippedGradient.ElementwiseMultiply(new Tensor(clippedGradient.Shape, PradTools.Cast(this.LearningRate)))));
                newOps.Add(newOp);
            }

            return newOps;
        }

        private PradResult ConcatMap(
            PradOp b,
            PradOp a,
            Func<PradOp, PradOp, int, object, int> mapFunc,
            object mappingObj)
        {
            var aShape = (int[])a.CurrentShape.Clone();
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

            var flattenedIndices = indices.Reshape(new int[] { indices.Shape[0] * indices.Shape[1] });

            var resortedFlatA = flattenedA.PradOp.Gather(flattenedIndices);
            var resortedA = resortedFlatA.PradOp.Reshape(aShape);

            return b.Concat(new[] { resortedA.Result }, axis: 1);
        }

        private (int, int)[] GeneratePairs(int[] indices)
        {
            var pairs = new List<(int, int)>();
            for (int i = 0; i < indices.Length; i++)
            {
                for (int j = i + 1; j < indices.Length; j++)
                {
                    pairs.Add((indices[i], indices[j]));
                }
            }

            return pairs.ToArray();
        }
    }
}
