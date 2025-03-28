//------------------------------------------------------------------------------
// <copyright file="PradOpExtensions.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.Extensions
{
    using System.Linq;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Extension methods for PradOp.
    /// </summary>
    public static class PradOpExtensions
    {
        /// <summary>
        /// Applies the SymmetricSoftmax transformation to the current tensor.
        /// SymmetricSoftmax = (x_i / (1 + |x_i|)) / (sum of (x_j / (1 + |x_j|)))
        /// Preserves signs while ensuring outputs sum to 1.
        /// </summary>
        /// <param name="pradOp">The PradOp instance containing the input tensor.</param>
        /// <param name="temperature">Temperature parameter for controlling the distribution sharpness.</param>
        /// <param name="usePowerTransform">Whether to use power transformation for temperature scaling.</param>
        /// <param name="axis">The axis along which to perform the softmax operation (default: 1).</param>
        /// <returns>The result of the SymmetricSoftmax operation.</returns>
        public static PradResult SymmetricSoftmax(this PradOp pradOp, double temperature = 1.0, bool usePowerTransform = true, int axis = 1)
        {
            // Step 1: Apply the x/(1+|x|) transformation to bound values to [-1, 1]
            var pradOpBranch = pradOp.Branch();
            var absX = pradOp.Abs();
            var one = new Tensor(pradOp.CurrentShape, PradTools.One);
            PradOp oneOp = new PradOp(one);
            var denominator = absX.Then(PradOp.AddOp, oneOp.CurrentTensor);
            var intermediates = denominator.PradOp.DivInto(pradOpBranch.CurrentTensor);

            // Create branches to track intermediate values for complex computation graph
            PradOp? intermediatesBranch = usePowerTransform ? intermediates.PradOp.Branch() : null;

            // Step 2: Apply temperature scaling
            PradResult temperatureAdjusted;
            if (usePowerTransform)
            {
                // Power transformation: sign(x) * |x|^(1/T)
                var absIntermediate = intermediates.PradOp.Abs();
                var powerResult = absIntermediate.PradOp.Pow(new Tensor(pradOp.CurrentShape, PradTools.One / PradTools.Cast(temperature)));

                // Get the signs from original intermediates
                var zero1 = new Tensor(intermediatesBranch!.CurrentShape, PradTools.Zero);
                var isNegative1 = intermediatesBranch.LessThan(zero1);
                var negativeOnes = new Tensor(intermediatesBranch.CurrentShape, PradTools.NegativeOne);
                var positiveOnes = new Tensor(intermediatesBranch.CurrentShape, PradTools.One);
                PradOp negativeOnesOp = new PradOp(negativeOnes);
                PradOp positiveOnesOp = new PradOp(positiveOnes);
                var signs = negativeOnesOp.Where(isNegative1.Result, positiveOnesOp.CurrentTensor);

                // Apply signs to power results
                temperatureAdjusted = powerResult.PradOp.Mul(signs.Result);
            }
            else
            {
                // Simple scaling: x/T
                var tempTensor = new Tensor(pradOp.CurrentShape, PradTools.Cast(temperature));
                temperatureAdjusted = intermediates.PradOp.Div(tempTensor);
            }

            // Step 3: Calculate total magnitude for normalization
            // First, separate positive and negative values
            var tempAdjBranches = temperatureAdjusted.PradOp.BranchStack(3);
            var tempAdjBranch1 = tempAdjBranches.Pop();
            var tempAdjBranch2 = tempAdjBranches.Pop();
            var tempAdjBranch3 = tempAdjBranches.Pop();
            var zero = new Tensor(tempAdjBranch1.CurrentShape, PradTools.Zero);
            var isNegative = tempAdjBranch1.LessThan(zero);

            // Create masks for positive and negative values
            var zeroTensor = new Tensor(tempAdjBranch1.CurrentShape, PradTools.Zero);
            PradOp zeroTensorOp = new PradOp(zeroTensor);

            // Get positive values
            var isPositive = isNegative.PradOp.SubFrom(new Tensor(isNegative.Result.Shape, PradTools.One));
            var positiveMask = tempAdjBranch2.Where(isPositive.Result, zeroTensorOp.CurrentTensor);

            // Get negative values (as absolute values)
            var negativeValues = tempAdjBranch3.Where(isNegative.Result, zeroTensorOp.CurrentTensor);
            var absNegativeValues = negativeValues.PradOp.Abs();

            // Sum positive and negative magnitudes along specified axis
            var sumPositive = positiveMask.PradOp.Sum(new[] { axis });
            var sumNegative = absNegativeValues.PradOp.Sum(new[] { axis });

            // Total magnitude is sum of positive and negative magnitudes
            var totalMagnitude = sumPositive.PradOp.Add(sumNegative.Result);

            // Expand dimensions to match original shape
            int[] broadcastShape = temperatureAdjusted.Result.Shape.ToArray();
            var totalMagnitudeBroadcast = totalMagnitude.PradOp.BroadcastTo(broadcastShape);

            // Step 4: Normalize by dividing by total magnitude
            var softmaxOutput = temperatureAdjusted.PradOp.Div(totalMagnitudeBroadcast.Result);

            var sobranch1 = softmaxOutput.Branch();

            var sShape = softmaxOutput.PradOp.CurrentShape;
            var columnShape = PradTools.Cast(sShape[^1]);

            var sumOverColumns = sobranch1.Sum(new int[] { axis });

            var oneT = new Tensor(sumOverColumns.PradOp.CurrentShape, PradTools.One);
            var oneTOp = new PradOp(oneT);

            var diff = oneTOp.Sub(sumOverColumns.Result);
            var columns = new Tensor(sumOverColumns.PradOp.CurrentShape, columnShape);
            var columnsOp = new PradOp(columns);
            var adder = diff.PradOp.Div(columnsOp.CurrentTensor).PradOp.BroadcastTo(sShape);

            var softmaxOutputAdjusted = softmaxOutput.PradOp.Add(adder.Result);

            return softmaxOutputAdjusted;
        }

        /// <summary>
        /// Applies Gaussian noise to tensor values.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdDev">The standard deviation.</param>
        /// <returns>The PRAD result.</returns>
        public static PradResult GaussianNoise(this PradOp pradOp, double mean = 0.0, double stdDev = 0.1)
        {
            var noise = Tensor.RandomUniform(pradOp.CurrentShape, PradTools.Cast(mean - stdDev), PradTools.Cast(mean + stdDev));
            return pradOp.Add(noise);
        }

        /// <summary>
        /// Apply dropout with the specified probability.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <param name="dropProb">The probability.</param>
        /// <returns>The PRAD result.</returns>
        public static PradResult Dropout(this PradOp pradOp, double dropProb = 0.5)
        {
            var mask = Tensor.RandomUniform(pradOp.CurrentShape);
            var dropMask = mask > PradTools.Cast(dropProb);
            var scale = new Tensor(pradOp.CurrentShape, PradTools.One / (PradTools.One - PradTools.Cast(dropProb)));
            return pradOp.Mul(dropMask).Then(PradOp.MulOp, scale);
        }

        /// <summary>
        /// Applies Gated Linear Unit (GLU) activation.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <returns>The PRAD result.</returns>
        public static PradResult GLU(this PradOp pradOp)
        {
            var shape = pradOp.CurrentShape;
            var splitSize = shape[^1] / 2;

            var branch = pradOp.Branch();

            // Split input into two halves
            var input = pradOp.Indexer(":", $":{splitSize}");
            var gates = branch.Indexer(":", $"{splitSize}:");

            // Apply sigmoid to gates
            var activated = gates.PradOp.Sigmoid();

            // Multiply input by activated gates
            return input.Then(PradOp.MulOp, activated.Result);
        }

        /// <summary>
        /// Computes the sigmoid function 1/(1 + e^(-x)) using primitive operations.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <returns>A PRAD result.</returns>
        public static PradResult Sigmoid(this PradOp pradOp)
        {
            // Create -x
            var negInput = pradOp.Mul(new Tensor(pradOp.CurrentShape, PradTools.NegativeOne));

            // Calculate e^(-x)
            var expNegInput = negInput.Then(PradOp.ExpOp);

            // Add 1 to get (1 + e^(-x))
            var oneTensor = new Tensor(pradOp.CurrentShape, PradTools.One);
            var denominator = expNegInput.Then(PradOp.AddOp, oneTensor);

            // Compute 1/(1 + e^(-x))
            return denominator.PradOp.DivInto(oneTensor);
        }

        /// <summary>
        /// Applies layer normalization to the input tensor.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <param name="epsilon">Epsilon.</param>
        /// <returns>A PRAD result.</returns>
        public static PradResult LayerNorm(this PradOp pradOp, double epsilon = 1e-5)
        {
            var branch = pradOp.Branch();

            // Calculate mean along last dimension
            var mean = pradOp.Mean(-1);

            // Calculate variance
            var meanBroadcast = mean.Then(PradOp.BroadcastToOp, branch.CurrentShape);
            var centered = meanBroadcast.PradOp.SubFrom(branch.CurrentTensor);
            var centeredBranch = centered.Branch();
            var squared = centered.Then(PradOp.SquareOp);
            var variance = squared.Then(PradOp.MeanOp, -1);

            // Normalize
            var varianceBroadcast = variance.Then(PradOp.BroadcastToOp, branch.CurrentShape);
            var stddev = varianceBroadcast.Then(x => x.PradOp.Add(new Tensor(x.PradOp.CurrentShape, PradTools.Cast(epsilon))))
                                        .Then(PradOp.SquareRootOp);

            return stddev.PradOp.DivInto(centeredBranch.CurrentTensor);
        }

        /// <summary>
        /// Applies FFT-based spectral regularization to encourage specific frequency characteristics.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <param name="freqWeights">The frequency weights.</param>
        /// <returns>The PRAD result.</returns>
        public static PradResult SpectralRegularize(this PradOp pradOp, PradOp freqWeights)
        {
            // Square the values for power spectrum
            var squared = pradOp.Square();

            // Apply frequency weighting
            var weighted = squared.Then(PradOp.MulOp, freqWeights.CurrentTensor);

            // Sum across all dimensions
            return weighted.Then(PradOp.SumOp, new[] { 0, 1 });
        }

        /// <summary>
        /// Applies total variation regularization to encourage smoothness.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <returns>The PRAD result.</returns>
        public static PradResult TotalVariation(this PradOp pradOp)
        {
            // Calculate differences in x and y directions
            var branches = pradOp.BranchStack(3);
            var left = branches.Pop().Indexer(":", ":-1");
            var right = pradOp.Indexer(":", "1:");
            var xDiff = right.PradOp.Sub(left.Result);

            var top = branches.Pop().Indexer(":-1", ":");
            var bottom = branches.Pop().Indexer("1:", ":");
            var yDiff = bottom.PradOp.Sub(top.Result);

            // Sum absolute differences
            var xLoss = xDiff.Then(PradOp.AbsOp).Then(PradOp.SumOp, new[] { 0, 1 });
            var yLoss = yDiff.Then(PradOp.AbsOp).Then(PradOp.SumOp, new[] { 0, 1 });

            return xLoss.Then(PradOp.AddOp, yLoss.Result);
        }

        /// <summary>
        /// Applies local response normalization across feature maps.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <param name="localSize">The local size.</param>
        /// <param name="alpha">Alpha.</param>
        /// <param name="beta">Beta.</param>
        /// <returns>A PRAD result.</returns>
        public static PradResult LocalResponseNorm(this PradOp pradOp, int localSize = 5, double alpha = 1e-4, double beta = 0.75)
        {
            var branch = pradOp.Branch();

            // Square the input
            var squared = pradOp.Square();

            // Calculate sum of squares in local neighborhood
            var sumSquares = squared.Then(PradOp.SumOp, new[] { -1 });
            var localNorm = sumSquares.Then(x =>
            {
                var scaled = x.PradOp.Mul(new Tensor(x.PradOp.CurrentShape, PradTools.Cast(alpha)));
                return scaled.Then(PradOp.AddOp, new Tensor(scaled.PradOp.CurrentShape, PradTools.One))
                           .Then(y => y.PradOp.Pow(new Tensor(y.PradOp.CurrentShape, PradTools.Cast(-beta))));
            });

            // Apply normalization
            return localNorm.PradOp.Mul(branch.CurrentTensor);
        }

        /// <summary>
        /// Applies batch normalization to the input tensor.
        /// </summary>
        /// <param name="pradOp">PradOp.</param>
        /// <param name="gamma">Gamme.</param>
        /// <param name="beta">Beta.</param>
        /// <param name="epsilon">Epsilon.</param>
        /// <returns>A PRAD result.</returns>
        public static PradResult BatchNorm(this PradOp pradOp, PradOp gamma, PradOp beta, double epsilon = 1e-5)
        {
            var branch = pradOp.Branch();

            // Calculate mean and variance across batch dimension
            var mean = pradOp.Mean(0);
            var meanBroadcast = mean.Then(PradOp.BroadcastToOp, branch.CurrentShape);
            var centered = meanBroadcast.PradOp.SubFrom(branch.CurrentTensor);

            var centeredBranch = centered.Branch();

            var squared = centered.Then(PradOp.SquareOp);
            var variance = squared.Then(PradOp.MeanOp, 0);

            // Normalize
            var varianceBroadcast = variance.Then(PradOp.BroadcastToOp, branch.CurrentShape);
            var stddev = varianceBroadcast.Then(x => x.PradOp.Add(new Tensor(x.PradOp.CurrentShape, PradTools.Cast(epsilon))))
                                        .Then(PradOp.SquareRootOp);

            var normalized = stddev.PradOp.DivInto(centeredBranch.CurrentTensor);

            // Scale and shift
            var scaled = normalized.Then(PradOp.MulOp, gamma.CurrentTensor);
            return scaled.Then(PradOp.AddOp, beta.CurrentTensor);
        }
    }
}