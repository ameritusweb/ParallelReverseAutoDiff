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
        /// Applies noisy scaling and rotation to a vector field using the reparameterization trick.
        /// </summary>
        /// <param name="pradOp">The PradOp instance containing the vector field (magnitudes left, angles right).</param>
        /// <param name="scaleFactors">Tensor of scale factors for magnitudes.</param>
        /// <param name="rotationFactors">Tensor of rotation factors [0,1] where 1 means 180° clockwise rotation.</param>
        /// <param name="scaleMean">Mean tensor for scale noise.</param>
        /// <param name="scaleLogVar">Log variance tensor for scale noise.</param>
        /// <param name="rotationMean">Mean tensor for rotation noise.</param>
        /// <param name="rotationLogVar">Log variance tensor for rotation noise.</param>
        /// <returns>A tensor with noisy scaled magnitudes and rotated angles.</returns>
        public static PradResult NoisyScaleAndRotate(
            this PradOp pradOp,
            PradOp scaleFactors,
            PradOp rotationFactors,
            PradOp scaleMean,
            PradOp scaleLogVar,
            PradOp rotationMean,
            PradOp rotationLogVar)
        {
            // Split input into magnitude and angle components
            var shape = pradOp.CurrentShape;
            var halfCols = shape[^1] / 2;

            var pradOpBranch = pradOp.Branch();
            var magnitudes = pradOp.Indexer(":", $":{halfCols}");
            var angles = pradOpBranch.Indexer(":", $"{halfCols}:");

            // Generate random noise using reparameterization trick for scaling
            var scaleEpsilon = Tensor.RandomNormal(scaleMean.CurrentShape);
            var scaleEpsilonOp = new PradOp(scaleEpsilon);

            var scaleStdDev = scaleLogVar.Mul(new Tensor(scaleLogVar.CurrentShape, PradTools.Half))
                                               .PradOp.Exp();

            var scaleNoise = scaleStdDev.PradOp.Mul(scaleEpsilonOp.CurrentTensor)
                                              .PradOp.Add(scaleMean.CurrentTensor);

            // Generate random noise using reparameterization trick for rotation
            var rotationEpsilon = Tensor.RandomNormal(rotationMean.CurrentShape);
            var rotationEpsilonOp = new PradOp(rotationEpsilon);

            var rotationStdDev = rotationLogVar.Mul(new Tensor(rotationLogVar.CurrentShape, PradTools.Half))
                                                     .PradOp.Exp();

            var rotationNoise = rotationStdDev.PradOp.Mul(rotationEpsilonOp.CurrentTensor)
                                                    .PradOp.Add(rotationMean.CurrentTensor);

            // Apply noisy scaling to magnitudes
            var noisyScaleFactors = scaleFactors.Add(scaleNoise.Result);
            var scaledMagnitudes = magnitudes.PradOp.Mul(noisyScaleFactors.Result);

            // Apply noisy rotation to angles
            // Convert rotation factor [0,1] to radians [0,π]
            var pi = new Tensor(rotationFactors.CurrentShape, PradMath.PI);
            var rotationRadians = rotationFactors.Mul(pi);

            // Add noise to rotation
            var noisyRotationFactors = rotationRadians.PradOp.Add(rotationNoise.Result);

            // Apply rotation
            var rotatedAngles = angles.PradOp.Add(noisyRotationFactors.Result);

            // Ensure angles stay within [-2π,2π]
            var twoPi = new Tensor(rotatedAngles.PradOp.CurrentShape, PradTools.Two * PradMath.PI);
            var normalizedAngles = rotatedAngles.PradOp.Modulus(twoPi);

            // Concatenate scaled magnitudes and rotated angles
            return scaledMagnitudes.PradOp.Concat(new[] { normalizedAngles.Result }, axis: -1);
        }

        /// <summary>
        /// Computes the alignment field for a vector field represented in magnitude-angle format.
        /// The alignment score measures how consistently vectors point in the same direction within local neighborhoods.
        /// </summary>
        /// <param name="pradOp">The PradOp instance containing the vector field (magnitudes left, angles right).</param>
        /// <param name="windowSize">Size of the window for local averaging (default: 5).</param>
        /// <param name="sigma">Standard deviation for Gaussian kernel (default: 1.0).</param>
        /// <returns>A tensor containing the alignment field where values close to 1.0 indicate high alignment.
        /// and values close to 0.0 indicate scattered directions.</returns>
        public static PradResult ComputeAlignmentField(this PradOp pradOp, int windowSize = 5, double sigma = 1.0)
        {
            // Step 1: Extract angle field
            var shape = pradOp.CurrentShape;
            var halfCols = shape[^1] / 2;
            var angles = pradOp.Indexer(":", $"{halfCols}:");

            // Step 2: Convert to unit vectors
            var anglesBranch = angles.Branch();
            var ux = angles.PradOp.Cos();  // x-component of unit vector
            var uy = anglesBranch.Sin();   // y-component of unit vector

            // Create Gaussian kernel for local averaging
            var gaussian = CreateGaussianKernel(windowSize, sigma);
            var gaussianTensor = new Tensor(new[] { 1, 1, 1, windowSize * windowSize }, gaussian);
            var kernelOp = new PradOp(gaussianTensor);

            // Step 3: Compute local averages of unit vectors
            var uxBranch = ux.Branch();
            var uyBranch = uy.Branch();

            // Average x-components
            var uxAvg = ux.PradOp.ExtractPatches(
                    new[] { windowSize, windowSize },
                    new[] { 1, 1 },
                    "SAME")
                .PradOp.Mul(kernelOp.CurrentTensor)
                .PradOp.Sum(new[] { -1 });

            // Average y-components
            var uyAvg = uy.PradOp.ExtractPatches(
                    new[] { windowSize, windowSize },
                    new[] { 1, 1 },
                    "SAME")
                .PradOp.Mul(kernelOp.CurrentTensor)
                .PradOp.Sum(new[] { -1 });

            // Step 4: Compute alignment score
            // A(x,y) = sqrt(ux_avg^2 + uy_avg^2)
            var uxAvgSquared = uxAvg.PradOp.Square();
            var uyAvgSquared = uyAvg.PradOp.Square();

            var sumSquares = uxAvgSquared.PradOp.Add(uyAvgSquared.Result);
            var alignment = sumSquares.PradOp.SquareRoot();

            // Optional: Apply normalization to ensure output is in [0,1]
            // This step might not be necessary as the norm of averaged unit vectors.
            // should already be <= 1, but it helps handle numerical precision issues
            var epsilon = new Tensor(alignment.PradOp.CurrentShape, 1e-10f);
            var alignmentNormalized = alignment.PradOp.Clip(0.0, 1.0);

            return alignmentNormalized;
        }

        /// <summary>
        /// Computes the curvature field for a vector field represented in magnitude-angle format.
        /// The curvature is calculated as the Frobenius norm of the Jacobian of the unit vector field.
        /// </summary>
        /// <param name="pradOp">The PradOp instance containing the vector field (magnitudes left, angles right).</param>
        /// <returns>A tensor containing the curvature field where higher values indicate stronger directional changes.</returns>
        public static PradResult ComputeCurvatureField(this PradOp pradOp)
        {
            // Step 1: Split into magnitude and angle components and convert to unit vectors
            var shape = pradOp.CurrentShape;
            var halfCols = shape[^1] / 2;

            var angles = pradOp.Indexer(":", $"{halfCols}:");

            // Convert to unit vectors (cos θ, sin θ)
            var anglesBranch = angles.Branch();
            var uField = angles.PradOp.Cos();  // u = cos(θ)
            var vField = anglesBranch.Sin();   // v = sin(θ)

            // Step 2: Create Sobel kernels for gradient computation
            var sobelX = new Tensor(
                new[] { 1, 1, 1, 3 * 3 },
                new double[] { -1, 0, 1, -2, 0, 2, -1, 0, 1 });
            var sobelY = new Tensor(
                new[] { 1, 1, 1, 3 * 3 },
                new double[] { -1, -2, -1, 0,  0,  0, 1,  2,  1 });

            // Normalize Sobel kernels
            var normFactor = new Tensor(sobelX.Shape, 1.0f / 8.0f);
            var sobelXOp = new PradOp(sobelX.ElementwiseMultiply(normFactor));
            var sobelYOp = new PradOp(sobelY.ElementwiseMultiply(normFactor));

            // Compute partial derivatives using convolution with Sobel kernels
            var uFieldBranch = uField.BranchStack(2);
            var vFieldBranch = vField.BranchStack(2);

            // ∂u/∂x
            var dudx = uField.PradOp.ExtractPatches(new[] { 3, 3 }, new[] { 1, 1 }, "SAME")
                             .PradOp.Mul(sobelXOp.CurrentTensor)
                             .PradOp.Sum(new[] { -1 });

            // ∂u/∂y
            var dudy = uFieldBranch.Pop().ExtractPatches(new[] { 3, 3 }, new[] { 1, 1 }, "SAME")
                                  .PradOp.Mul(sobelYOp.CurrentTensor)
                                  .PradOp.Sum(new[] { -1 });

            // ∂v/∂x
            var dvdx = vField.PradOp.ExtractPatches(new[] { 3, 3 }, new[] { 1, 1 }, "SAME")
                             .PradOp.Mul(sobelXOp.CurrentTensor)
                             .PradOp.Sum(new[] { -1 });

            // ∂v/∂y
            var dvdy = vFieldBranch.Pop().ExtractPatches(new[] { 3, 3 }, new[] { 1, 1 }, "SAME")
                                  .PradOp.Mul(sobelYOp.CurrentTensor)
                                  .PradOp.Sum(new[] { -1 });

            // Step 3: Compute Frobenius norm of the Jacobian
            // κ = sqrt((∂u/∂x)² + (∂u/∂y)² + (∂v/∂x)² + (∂v/∂y)²)

            // Square all components
            var dudxBranch = dudx.Branch();
            var dudyBranch = dudy.Branch();
            var dvdxBranch = dvdx.Branch();

            var dudxSquared = dudx.PradOp.Square();
            var dudySquared = dudy.PradOp.Square();
            var dvdxSquared = dvdx.PradOp.Square();
            var dvdySquared = dvdy.PradOp.Square();

            // Sum squared components
            var sumSquares = dudxSquared.PradOp.Add(dudySquared.Result)
                                       .PradOp.Add(dvdxSquared.Result)
                                       .PradOp.Add(dvdySquared.Result);

            // Take square root to get curvature
            var curvature = sumSquares.PradOp.SquareRoot();

            var reshaped = curvature.PradOp.Reshape(new[] { 20, 20, 1 });

            // Optional: Apply smoothing to reduce noise
            var gaussianKernel = CreateGaussianKernel(3, 1.0);
            var gaussianTensor = new Tensor(new[] { 1, 1, 1, 3 * 3 }, gaussianKernel);
            var kernelOp = new PradOp(gaussianTensor);

            var smoothedCurvature = curvature.PradOp.ExtractPatches(new[] { 3, 3 }, new[] { 1, 1 }, "SAME")
                                            .PradOp.Mul(kernelOp.CurrentTensor)
                                            .PradOp.Sum(new[] { -1 });

            return smoothedCurvature;
        }

        /// <summary>
        /// Computes the Structure Tensor Entropy field for a vector field represented in magnitude-angle format.
        /// The input tensor should have magnitudes in the left half and angles in the right half.
        /// </summary>
        /// <param name="pradOp">The PradOp instance containing the vector field.</param>
        /// <param name="windowSize">Size of the Gaussian window for local averaging (default: 5).</param>
        /// <param name="sigma">Standard deviation for Gaussian kernel (default: 1.0).</param>
        /// <returns>A tensor containing the entropy field where higher values indicate more isotropic flow.</returns>
        public static PradResult ComputeStructureTensorEntropy(this PradOp pradOp, int windowSize = 5, double sigma = 1.0)
        {
            // Step 0: Split into magnitude and angle components
            var shape = pradOp.CurrentShape;
            var halfCols = shape[^1] / 2;

            var pradOpBranch = pradOp.Branch();
            var magnitudes = pradOp.Indexer(":", $":{halfCols}");
            var angles = pradOpBranch.Indexer(":", $"{halfCols}:");

            // Convert to Cartesian coordinates
            var anglesBranch = angles.Branch();
            var cosAngles = angles.PradOp.Cos();
            var sinAngles = anglesBranch.Sin();

            var magnitudesBranch = magnitudes.Branch();
            var vx = magnitudes.PradOp.Mul(cosAngles.Result);
            var vy = magnitudesBranch.Mul(sinAngles.Result);

            // Step 1: Compute structure tensor components
            var vxBranch = vx.Branch();
            var vyBranch = vy.Branch();

            // Compute vector products
            var vxx = vx.PradOp.Square();  // vx^2
            var vyy = vy.PradOp.Mul(vyBranch.CurrentTensor);  // vy^2
            var vxy = vxBranch.Mul(vy.Result);               // vx*vy

            // Create Gaussian kernel for local averaging
            var gaussian = CreateGaussianKernel(windowSize, sigma);
            var gaussianTensor = new Tensor(new[] { 1, 1, 1, windowSize * windowSize }, gaussian);
            var kernelOp = new PradOp(gaussianTensor);

            // Apply local averaging using convolution
            var vxxSmoothed = vxx.PradOp.ExtractPatches(new[] { windowSize, windowSize }, new[] { 1, 1 }, "SAME")
                                 .PradOp.Mul(kernelOp.CurrentTensor)
                                 .PradOp.Sum(new[] { -1 });

            var vyySmoothed = vyy.PradOp.ExtractPatches(new[] { windowSize, windowSize }, new[] { 1, 1 }, "SAME")
                                 .PradOp.Mul(kernelOp.CurrentTensor)
                                 .PradOp.Sum(new[] { -1 });

            var vxySmoothed = vxy.PradOp.ExtractPatches(new[] { windowSize, windowSize }, new[] { 1, 1 }, "SAME")
                                 .PradOp.Mul(kernelOp.CurrentTensor)
                                 .PradOp.Sum(new[] { -1 });

            // Step 2: Compute eigenvalues
            // For a 2x2 matrix, eigenvalues can be computed directly
            var vxxBranch = vxxSmoothed.Branch();
            var vyyBranch = vyySmoothed.Branch();
            var vxyBranch = vxySmoothed.Branch();

            // Trace and determinant
            var trace = vxxSmoothed.PradOp.Add(vyySmoothed.Result);
            var det = vxxBranch.Mul(vyyBranch.CurrentTensor)
                              .PradOp.Sub(vxyBranch.Mul(vxySmoothed.Result).Result);

            var traceBranch = trace.BranchStack(2);
            var detBranch = det.Branch();

            // Compute discriminant
            var discriminant = trace.PradOp.Square()
                                   .PradOp.Sub(det.PradOp.Mul(new Tensor(det.PradOp.CurrentShape, 4.0f)).Result)
                                   .PradOp.SquareRoot();

            // Compute eigenvalues
            var lambda1 = traceBranch.Pop().Add(discriminant.Result)
                                    .PradOp.Mul(new Tensor(trace.PradOp.CurrentShape, 0.5f));

            var lambda2 = traceBranch.Pop().Sub(discriminant.Result)
                               .PradOp.Mul(new Tensor(trace.PradOp.CurrentShape, 0.5f));

            // Step 3: Normalize eigenvalues
            var lambda1Branch = lambda1.Branch();
            var lambda2Branch = lambda2.Branch();

            var sumEigenvalues = lambda1.PradOp.Add(lambda2.Result);
            var epsilon = new Tensor(sumEigenvalues.PradOp.CurrentShape, 1e-10f);
            var denominator = sumEigenvalues.PradOp.Add(epsilon);

            var p1 = lambda1Branch.Div(denominator.Result);
            var p2 = lambda2Branch.Div(denominator.Result);

            // Step 4: Compute Shannon entropy
            var p1Branch = p1.Branch();
            var p2Branch = p2.Branch();

            // Add small epsilon to avoid log(0)
            var p1Safe = p1.PradOp.Add(epsilon);
            var p2Safe = p2.PradOp.Add(epsilon);

            var ln2 = new Tensor(p1.PradOp.CurrentShape, (float)PradMath.Log(2));
            var entropy = p1Branch.Mul(p1Safe.PradOp.Log().Result.ElementwiseDivide(ln2))
                                 .PradOp.Add(p2Branch.Mul(p2Safe.PradOp.Log().Result.ElementwiseDivide(ln2)).Result)
                                 .PradOp.Mul(new Tensor(p1.PradOp.CurrentShape, -1.0f));

            return entropy;
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
        /// Applies the GELU (Gaussian Error Linear Unit) activation function.
        /// GELU(x) = x * Φ(x) where Φ(x) is the CDF of the standard normal distribution.
        /// Uses the approximation: GELU(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³))).
        /// </summary>
        /// <param name="pradOp">The PradOp instance containing the input tensor.</param>
        /// <returns>The result of the GELU operation.</returns>
        public static PradResult GELU(this PradOp pradOp)
        {
            // Constants for GELU approximation
            var sqrt2ByPi = PradMath.Sqrt(PradTools.Two / PradMath.PI);
            var constant = new Tensor(pradOp.CurrentShape, PradTools.Cast(sqrt2ByPi));
            var coef = new Tensor(pradOp.CurrentShape, 0.044715f);

            // Create branches for reusing the input
            var inputBranch1 = pradOp.Branch();
            var inputBranch2 = pradOp.Branch();

            // Calculate x³
            var xCubed = inputBranch1.Mul(pradOp.CurrentTensor)
                                    .Then(PradOp.MulOp, pradOp.CurrentTensor);

            // Calculate (x + 0.044715x³)
            var innerSum = xCubed.Then(PradOp.MulOp, coef)
                                .Then(result => result.PradOp.Add(inputBranch2.CurrentTensor));

            // Calculate √(2/π) * (x + 0.044715x³)
            var scaled = innerSum.Then(PradOp.MulOp, constant);

            // Calculate tanh(√(2/π) * (x + 0.044715x³))
            var tanhResult = scaled.Then(PradOp.TanhOp);

            // Calculate (1 + tanh(...))
            var oneTensor = new Tensor(pradOp.CurrentShape, PradTools.One);
            var tanhPlusOne = tanhResult.Then(PradOp.AddOp, oneTensor);

            // Calculate 0.5x * (1 + tanh(...))
            var halfTensor = new Tensor(pradOp.CurrentShape, PradTools.Half);
            var halfX = pradOp.Mul(halfTensor);

            return halfX.Then(PradOp.MulOp, tanhPlusOne.Result);
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

        /// <summary>
        /// Creates a 2D Gaussian kernel for local averaging.
        /// </summary>
        private static double[] CreateGaussianKernel(int size, double sigma)
        {
            var kernel = new double[size * size];
            var center = size / 2;
            var sum = 0.0d;

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    var dx = x - center;
                    var dy = y - center;
                    var g = PradMath.Exp(-((dx * dx) + (dy * dy)) / (2 * sigma * sigma));
                    kernel[(y * size) + x] = g;
                    sum += g;
                }
            }

            // Normalize kernel
            for (int i = 0; i < kernel.Length; i++)
            {
                kernel[i] /= sum;
            }

            return kernel;
        }
    }
}