//------------------------------------------------------------------------------
// <copyright file="PradOpExtensions.Loss.cs" author="ameritusweb" date="3/20/2025">
// Copyright (c) 2025 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.Extensions
{
    using System;
    using System.Linq;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Extension methods for PradOp.
    /// </summary>
    public static partial class PradOpExtensions
    {
        /// <summary>
        /// Computes the Mean Squared Error (MSE) loss between predictions and targets along specified axes.
        /// MSE = (1/n) * Σ(y - ŷ)².
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target values.</param>
        /// <param name="axes">The axes.</param>
        /// <returns>A scalar tensor containing the MSE loss.</returns>
        public static PradResult MeanSquaredError(this PradOp predictions, PradOp targets, int[]? axes = null)
        {
            var shape = predictions.CurrentShape;

            // If axes not specified, compute along all axes
            if (axes == null)
            {
                axes = Enumerable.Range(0, shape.Length).ToArray();
            }

            // Calculate size of remaining dimensions and number of elements to average over
            var size = shape.Length - axes.Length;
            int numberOfElements = 1;
            for (int i = 0; i < axes.Length; ++i)
            {
                numberOfElements *= shape[axes[i]];
            }

            // Create new shape for the output
            var newShape = shape.Take(size).Concat(new int[] { 1 }).ToArray();
            var divT = new Tensor(newShape, PradTools.Cast(numberOfElements));

            // Compute MSE
            var diff = predictions.Sub(targets.CurrentTensor);
            var squared = diff.Then(PradOp.SquareOp);
            var sum = squared.PradOp.Sum(axes);
            var avg = sum.PradOp.Div(divT);

            return avg;
        }

        /// <summary>
        /// Computes the Mean Squared Error (MSE) loss between predictions and targets with specific reduction mode.
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target values.</param>
        /// <param name="reduction">The axes.</param>
        /// <returns>A scalar tensor containing the MSE loss.</returns>
        public static PradResult MeanSquaredError(this PradOp predictions, PradOp targets, string reduction = "mean")
        {
            var diff = predictions.Sub(targets.CurrentTensor);
            var squared = diff.Then(PradOp.SquareOp);

            switch (reduction.ToLower())
            {
                case "none":
                    return squared;

                case "sum":
                    return squared.Then(PradOp.SumOp, Enumerable.Range(0, predictions.CurrentShape.Length).ToArray());

                case "mean":
                default:
                    var shape = predictions.CurrentShape;
                    var axes = Enumerable.Range(0, shape.Length).ToArray();
                    return MeanSquaredError(predictions, targets, axes);
            }
        }

        /// <summary>
        /// Computes the weighted Mean Squared Error loss between predictions and targets.
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target values.</param>
        /// <param name="weights">The weights.</param>
        /// <param name="axes">The axes.</param>
        /// <returns>A scalar tensor containing the MSE loss.</returns>
        public static PradResult WeightedMeanSquaredError(this PradOp predictions, PradOp targets, PradOp weights, int[]? axes = null)
        {
            var diff = predictions.Sub(targets.CurrentTensor);
            var squared = diff.Then(PradOp.SquareOp);
            var weighted = squared.Then(PradOp.MulOp, weights.CurrentTensor);

            if (axes == null)
            {
                axes = Enumerable.Range(0, predictions.CurrentShape.Length).ToArray();
            }

            var weightSum = weights.Sum(axes);
            var epsilon = new Tensor(weightSum.PradOp.CurrentShape, PradTools.Epsilon9);
            var safeDivisor = weightSum.Then(w => w.PradOp.Add(epsilon));

            var sum = weighted.PradOp.Sum(axes);
            return sum.Then(s => s.PradOp.Div(safeDivisor.Result));
        }

        /// <summary>
        /// Computes the Relative Mean Squared Error loss between predictions and targets.
        /// RMSE = MSE(predictions, targets) / MSE(targets, zeros).
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target values.</param>
        /// <param name="axes">The axes.</param>
        /// <returns>A scalar tensor containing the MSE loss.</returns>
        public static PradResult RelativeMeanSquaredError(this PradOp predictions, PradOp targets, int[]? axes = null)
        {
            var zeros = new Tensor(targets.CurrentShape);
            var mse = predictions.MeanSquaredError(targets, axes);
            var normalization = targets.MeanSquaredError(new PradOp(zeros), axes);

            var epsilon = new Tensor(normalization.PradOp.CurrentShape, PradTools.Epsilon9);
            var safeDivisor = normalization.Then(n => n.PradOp.Add(epsilon));

            return mse.Then(m => m.PradOp.Div(safeDivisor.Result));
        }

        /// <summary>
        /// Computes the Binary Cross Entropy loss between predictions and targets.
        /// BCE = -Σ(y * log(ŷ) + (1 - y) * log(1 - ŷ)).
        /// </summary>
        /// <param name="predictions">The predicted probabilities (should be between 0 and 1).</param>
        /// <param name="targets">The binary target values (0 or 1).</param>
        /// <param name="epsilon">Small constant to avoid log(0).</param>
        /// <returns>A scalar tensor containing the BCE loss.</returns>
        public static PradResult BinaryCrossEntropy(this PradOp predictions, PradOp targets, double epsilon = 1e-7)
        {
            // Clip predictions to prevent log(0)
            var clipped = predictions.Clip(epsilon, 1.0 - epsilon);

            var clippedB = clipped.Branch();

            // Calculate log terms
            var logP = clipped.Then(PradOp.LogOp);
            var oneMinusP = clippedB.SubFrom(new Tensor(clippedB.CurrentShape, PradTools.One));
            var logOneMinusP = oneMinusP.Then(PradOp.LogOp);

            // Calculate the two terms of BCE
            var term1 = logP.PradOp.Mul(targets.CurrentTensor);
            var oneMinusTargets = targets.SubFrom(new Tensor(targets.CurrentShape, PradTools.One));
            var term2 = oneMinusTargets.PradOp.Mul(logOneMinusP.Result);

            // Combine terms and take mean
            var combined = term1.PradOp.Add(term2.Result);
            var negated = combined.PradOp.Mul(new Tensor(combined.Result.Shape, PradTools.NegativeOne));

            return negated.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Categorical Cross Entropy loss between predictions and targets.
        /// CCE = -Σ(y * log(ŷ)).
        /// </summary>
        /// <param name="predictions">The predicted class probabilities (should sum to 1).</param>
        /// <param name="targets">The one-hot encoded target values.</param>
        /// <param name="epsilon">Small constant to avoid log(0).</param>
        /// <returns>A scalar tensor containing the CCE loss.</returns>
        public static PradResult CategoricalCrossEntropy(this PradOp predictions, PradOp targets, double epsilon = 1e-7)
        {
            // Clip predictions to prevent log(0)
            var clipped = predictions.Clip(epsilon, 1.0);

            // Calculate -log(p)
            var logP = clipped.Then(PradOp.LogOp);
            var negLogP = logP.Then(l =>
                l.PradOp.Mul(new Tensor(l.PradOp.CurrentShape, PradTools.NegativeOne)));

            // Multiply by targets and take mean
            var product = negLogP.PradOp.Mul(targets.CurrentTensor);
            return product.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Huber loss between predictions and targets.
        /// Huber loss combines the best properties of MSE and MAE.
        /// L(y, ŷ) = 0.5(y - ŷ)² if |y - ŷ| ≤ δ
        ///         = δ|y - ŷ| - 0.5δ² otherwise.
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target values.</param>
        /// <param name="delta">The threshold at which to switch from MSE to MAE.</param>
        /// <returns>A scalar tensor containing the Huber loss.</returns>
        public static PradResult HuberLoss(this PradOp predictions, PradOp targets, double delta = 1.0)
        {
            var diff = predictions.Sub(targets.CurrentTensor);
            var absDiff = diff.Then(PradOp.AbsOp);

            // Create delta tensor
            var deltaTensor = new Tensor(predictions.CurrentShape, PradTools.Cast(delta));

            // Calculate quadratic term (0.5 * diff²)
            var squared = diff.Then(PradOp.SquareOp);
            var quadratic = squared.Then(s =>
                s.PradOp.Mul(new Tensor(s.PradOp.CurrentShape, PradTools.Half)));

            // Calculate linear term (delta * |diff| - 0.5 * delta²)
            var linear = absDiff.Then(a => a.PradOp.Mul(deltaTensor));
            var deltaSquared = new Tensor(
                predictions.CurrentShape,
                PradTools.Cast(0.5 * delta * delta));
            var linearAdjusted = linear.Then(l => l.PradOp.Sub(deltaSquared));

            // Use where to select appropriate term based on |diff| <= delta
            var condition = absDiff.Then(a => a.PradOp.LessThan(deltaTensor));
            var result = quadratic.PradOp.Where(condition.Result, linearAdjusted.Result);

            return result.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Kullback-Leibler Divergence loss between predictions and targets.
        /// KLD = Σ(y * log(y/ŷ)).
        /// </summary>
        /// <param name="predictions">The predicted probability distribution.</param>
        /// <param name="targets">The target probability distribution.</param>
        /// <param name="epsilon">Small constant to avoid log(0).</param>
        /// <returns>A scalar tensor containing the KL divergence loss.</returns>
        public static PradResult KLDivergence(this PradOp predictions, PradOp targets, double epsilon = 1e-7)
        {
            // Clip predictions to prevent division by zero
            var clippedPreds = predictions.Clip(epsilon, 1.0);

            var targetsB = targets.Branch();

            // Calculate log(targets/predictions)
            var ratio = clippedPreds.PradOp.DivInto(targets.CurrentTensor);
            var logRatio = ratio.Then(PradOp.LogOp);

            // Multiply by targets and take mean
            var product = logRatio.PradOp.Mul(targetsB.CurrentTensor);
            return product.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Hinge loss for binary classification.
        /// L(y, ŷ) = max(0, 1 - y * ŷ).
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target values (-1 or 1).</param>
        /// <returns>A scalar tensor containing the hinge loss.</returns>
        public static PradResult HingeLoss(this PradOp predictions, PradOp targets)
        {
            var product = predictions.Mul(targets.CurrentTensor);
            var diff = product.Then(p =>
                p.PradOp.SubFrom(new Tensor(p.PradOp.CurrentShape, PradTools.One)));

            var zero = new Tensor(predictions.CurrentShape, PradTools.Zero);
            var result = diff.Then(PradOp.MaxOp, zero);

            return result.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Cosine Similarity loss between predictions and targets.
        /// CosineSimilarity = -Σ(y * ŷ)/(||y|| * ||ŷ||).
        /// </summary>
        /// <param name="predictions">The predicted values.</param>
        /// <param name="targets">The target values.</param>
        /// <param name="epsilon">Small constant to avoid division by zero.</param>
        /// <returns>A scalar tensor containing the cosine similarity loss.</returns>
        public static PradResult CosineSimilarityLoss(this PradOp predictions, PradOp targets, double epsilon = 1e-7)
        {
            var predictionsB = predictions.Branch();

            // Calculate dot product
            var dotProduct = predictions.Mul(targets.CurrentTensor);
            var sumDotProduct = dotProduct.Then(PradOp.SumOp, new[] { -1 });

            // Calculate L2 norms
            var predSquared = predictionsB.Square();
            var predNorm = predSquared.Then(PradOp.SumOp, new[] { -1 })
                                    .Then(PradOp.SquareRootOp);

            var targetSquared = targets.Square();
            var targetNorm = targetSquared.Then(PradOp.SumOp, new[] { -1 })
                                        .Then(PradOp.SquareRootOp);

            // Calculate denominator with epsilon for stability
            var normProduct = predNorm.Then(p => p.PradOp.Mul(targetNorm.Result));
            var denominator = normProduct.Then(n =>
                n.PradOp.Add(new Tensor(n.PradOp.CurrentShape, PradTools.Cast(epsilon))));

            // Calculate final result and negate (since we want to minimize negative cosine similarity)
            var similarity = sumDotProduct.Then(s => s.PradOp.Div(denominator.Result));
            return similarity.Then(s =>
                s.PradOp.Mul(new Tensor(s.PradOp.CurrentShape, PradTools.NegativeOne)));
        }

        /// <summary>
        /// Computes the Focal loss, a modified cross entropy that reduces the relative loss for well-classified examples.
        /// FL(pt) = -α(1-pt)ᵧ * log(pt).
        /// </summary>
        /// <param name="predictions">The predicted probabilities.</param>
        /// <param name="targets">The target values (0 or 1).</param>
        /// <param name="gamma">Focusing parameter (γ ≥ 0).</param>
        /// <param name="alpha">Weighting factor (α ∈ [0,1]).</param>
        /// <param name="epsilon">Small constant to avoid log(0).</param>
        /// <returns>A scalar tensor containing the focal loss.</returns>
        public static PradResult FocalLoss(
            this PradOp predictions,
            PradOp targets,
            double gamma = 2.0,
            double alpha = 0.25,
            double epsilon = 1e-7)
        {
            // Clip predictions for numerical stability
            var clipped = predictions.Clip(epsilon, 1.0 - epsilon);

            // Calculate pt (probability of true class)
            var pt = clipped.PradOp.Mul(targets.CurrentTensor);
            var oneMinusTargets = targets.SubFrom(new Tensor(targets.CurrentShape, PradTools.One));
            var oneMinusPreds = clipped.Then(p =>
                p.PradOp.SubFrom(new Tensor(p.PradOp.CurrentShape, PradTools.One)));
            var ptComplement = oneMinusTargets.Then(t => t.PradOp.Mul(oneMinusPreds.Result));
            pt.PradOp.Add(ptComplement.Result);

            var ptB = pt.Branch();

            // Calculate (1-pt)ᵧ
            var oneMinusPt = pt.Then(p =>
                p.PradOp.SubFrom(new Tensor(p.PradOp.CurrentShape, PradTools.One)));
            var focusingWeight = oneMinusPt.Then(p =>
                p.PradOp.Pow(new Tensor(p.PradOp.CurrentShape, PradTools.Cast(gamma))));

            // Calculate -log(pt)
            var logPt = ptB.Log();
            var negLogPt = logPt.Then(l =>
                l.PradOp.Mul(new Tensor(l.PradOp.CurrentShape, PradTools.NegativeOne)));

            // Apply alpha weighting
            var alphaTensor = new Tensor(predictions.CurrentShape, PradTools.Cast(alpha));
            var alphaWeight = targets.Mul(alphaTensor);
            var oneMinusAlpha = alphaTensor.ElementwiseSub(new Tensor(alphaTensor.Shape, PradTools.One));
            var alphaWeightComplement = oneMinusTargets.Then(t => t.PradOp.Mul(oneMinusAlpha));
            var combinedAlpha = alphaWeight.Then(a => a.PradOp.Add(alphaWeightComplement.Result));

            // Combine all terms
            var focalLoss = focusingWeight.PradOp.Mul(combinedAlpha.Result)
                .PradOp.Mul(negLogPt.Result);

            return focalLoss.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Triplet loss for learning embeddings.
        /// Loss = max(0, d(a,p) - d(a,n) + margin)
        /// where d(x,y) is the distance between x and y.
        /// </summary>
        /// <param name="anchor">The anchor embeddings.</param>
        /// <param name="positive">The positive embeddings.</param>
        /// <param name="negative">The negative embeddings.</param>
        /// <param name="margin">The margin to enforce between positive and negative pairs.</param>
        /// <returns>A scalar tensor containing the triplet loss.</returns>
        public static PradResult TripletLoss(this PradOp anchor, PradOp positive, PradOp negative, double margin = 1.0)
        {
            var anchorB = anchor.Branch();

            // Calculate pairwise distances (using squared L2 norm)
            var positiveDiff = anchor.Sub(positive.CurrentTensor);
            var negativeDiff = anchorB.Sub(negative.CurrentTensor);

            var positiveDistSquared = positiveDiff.Then(PradOp.SquareOp)
                                                 .Then(PradOp.SumOp, new[] { -1 });
            var negativeDistSquared = negativeDiff.Then(PradOp.SquareOp)
                                                 .Then(PradOp.SumOp, new[] { -1 });

            // Calculate loss with margin
            var diff = positiveDistSquared.Then(p =>
                p.PradOp.Sub(negativeDistSquared.Result));
            var withMargin = diff.Then(d =>
                d.PradOp.Add(new Tensor(d.PradOp.CurrentShape, PradTools.Cast(margin))));

            // Apply ReLU (max(0, x))
            var zero = new Tensor(withMargin.PradOp.CurrentShape, PradTools.Zero);
            var result = withMargin.Then(PradOp.MaxOp, zero);

            return result.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Contrastive loss for learning embeddings in a Siamese network architecture.
        /// For similar pairs (y=1): L = (1/2)(d)²
        /// For dissimilar pairs (y=0): L = (1/2)(max(0, margin - d))²
        /// where d is the Euclidean distance between embeddings.
        /// </summary>
        /// <param name="embedding1">The first embedding.</param>
        /// <param name="embedding2">The second embedding.</param>
        /// <param name="labels">Binary labels: 1 for similar pairs, 0 for dissimilar pairs.</param>
        /// <param name="margin">The margin for dissimilar pairs (should be > 0).</param>
        /// <returns>A scalar tensor containing the contrastive loss.</returns>
        public static PradResult ContrastiveLoss(this PradOp embedding1, PradOp embedding2, PradOp labels, double margin = 1.0)
        {
            // Calculate the Euclidean distance between embeddings
            var diff = embedding1.Sub(embedding2.CurrentTensor);
            var diffSquared = diff.Then(PradOp.SquareOp);
            var sumSquared = diffSquared.Then(PradOp.SumOp, new[] { -1 });
            var distances = sumSquared.Then(PradOp.SquareRootOp);

            // Square the distances for the similar pair term
            var distanceSquared = distances.Then(PradOp.SquareOp);
            var halfTensor = new Tensor(distanceSquared.PradOp.CurrentShape, PradTools.Half);
            var similarTerm = distanceSquared.Then(d => d.PradOp.Mul(halfTensor));

            // Calculate the dissimilar pair term: (max(0, margin - d))²
            var marginTensor = new Tensor(distances.PradOp.CurrentShape, PradTools.Cast(margin));
            var marginTensorOp = new PradOp(marginTensor);
            var marginDiff = marginTensorOp.Sub(distances.Result);
            var zero = new Tensor(marginDiff.PradOp.CurrentShape, PradTools.Zero);
            var maxTerm = marginDiff.Then(PradOp.MaxOp, zero);
            var maxTermSquared = maxTerm.Then(PradOp.SquareOp);
            var dissimilarTerm = maxTermSquared.Then(d => d.PradOp.Mul(halfTensor));

            var labelsB = labels.Branch();

            // Combine terms using labels
            // For similar pairs (label=1): use similarTerm
            // For dissimilar pairs (label=0): use dissimilarTerm
            var result = similarTerm.PradOp.Mul(labels.CurrentTensor);

            var oneMinusLabels = labelsB.SubFrom(new Tensor(labelsB.CurrentShape, PradTools.One));
            var dissimilarContribution = oneMinusLabels.Then(l =>
                l.PradOp.Mul(dissimilarTerm.Result));

            var combinedLoss = result.Then(r =>
                r.PradOp.Add(dissimilarContribution.Result));

            return combinedLoss.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Computes the Online Contrastive loss with hard negative mining.
        /// This variant of contrastive loss automatically selects hard negative examples
        /// within a batch to improve training efficiency.
        /// </summary>
        /// <param name="embeddings">The batch of embeddings.</param>
        /// <param name="labels">The labels for each embedding.</param>
        /// <param name="margin">The margin for dissimilar pairs.</param>
        /// <param name="hardestK">Number of hard negatives to select per positive.</param>
        /// <returns>A scalar tensor containing the online contrastive loss.</returns>
        public static PradResult OnlineContrastiveLoss(
            this PradOp embeddings,
            PradOp labels,
            double margin = 1.0,
            int hardestK = 3)
        {
            // Calculate pairwise distances between all embeddings in the batch
            var embeddingsBranch = embeddings.Branch();

            // Expand dimensions to prepare for broadcasting
            var expanded1 = embeddings.ExpandDims(1);
            var expanded2 = embeddingsBranch.ExpandDims(0);

            // Calculate squared differences
            var diff = expanded1.Then(e => e.PradOp.Sub(expanded2.Result));
            var squaredDiff = diff.Then(PradOp.SquareOp);

            // Sum along feature dimension to get pairwise distances
            var distances = squaredDiff.Then(PradOp.SumOp, new[] { -1 })
                                     .Then(PradOp.SquareRootOp);

            // Create label matrix for pairs
            var labelMatrix = labels.ExpandDims(1)
                                  .Then(l => l.PradOp.Equals(labels.ExpandDims(0).Result));

            // Calculate positive pair loss
            var positiveDistances = distances.Then(d => labelMatrix.PradOp.Where(labelMatrix.Result, d.Result));
            var positiveLoss = positiveDistances.Then(PradOp.SquareOp)
                                              .Then(p => p.PradOp.Mul(new Tensor(p.PradOp.CurrentShape, PradTools.Half)));

            // Calculate negative pair loss with hard mining
            var negativeMatrix = labelMatrix.Then(l =>
                l.PradOp.SubFrom(new Tensor(l.PradOp.CurrentShape, PradTools.One)));

            var marginTensor = new Tensor(distances.PradOp.CurrentShape, PradTools.Cast(margin));
            var marginTensorOp = new PradOp(marginTensor);
            var negativeDistances = distances.Then(d => negativeMatrix.PradOp.Where(negativeMatrix.Result, d.Result));

            // Select hardest K negative examples
            var marginDiff = marginTensorOp.Sub(negativeDistances.Result);
            var zero = new Tensor(marginDiff.PradOp.CurrentShape, PradTools.Zero);
            var maxTerm = marginDiff.Then(PradOp.MaxOp, zero);

            // Sort and select top K highest losses
            var sortedLosses = maxTerm.Then(PradOp.SquareOp)
                                    .Then(m => m.PradOp.Mul(new Tensor(m.PradOp.CurrentShape, PradTools.Half)))
                                    .Then(l => l.PradOp.TopK(hardestK));

            // Combine positive and negative losses
            var combinedLoss = positiveLoss.Then(p => p.PradOp.Add(sortedLosses.Result));

            return combinedLoss.Then(PradOp.MeanOp, -1);
        }

        /// <summary>
        /// Helper method to compute the top K values in a tensor.
        /// </summary>
        private static PradResult TopK(this PradOp input, int k)
        {
            // Sort values in descending order
            var sorted = input.CustomOperation(
                operation: tensor =>
                {
                    var result = new Tensor(new[] { tensor.Shape[0], k });
                    for (int i = 0; i < tensor.Shape[0]; i++)
                    {
                        var row = new double[tensor.Shape[1]];
                        for (int j = 0; j < tensor.Shape[1]; j++)
                        {
                            row[j] = tensor[i, j];
                        }

                        Array.Sort(row);
                        Array.Reverse(row);
                        for (int j = 0; j < k; j++)
                        {
                            result[i, j] = row[j];
                        }
                    }

                    return result;
                },
                reverseOperation: (input, output, upstreamGrad) =>
                {
                    var gradient = new Tensor(input.Shape);

                    // Gradient flows back through the k largest values
                    for (int i = 0; i < input.Shape[0]; i++)
                    {
                        var indices = new int[k];
                        var row = new double[input.Shape[1]];
                        for (int j = 0; j < input.Shape[1]; j++)
                        {
                            row[j] = input[i, j];
                        }

                        // Find indices of k largest values
                        for (int j = 0; j < k; j++)
                        {
                            double max = double.MinValue;
                            int maxIdx = -1;
                            for (int l = 0; l < row.Length; l++)
                            {
                                if (row[l] > max && !indices.Contains(l))
                                {
                                    max = row[l];
                                    maxIdx = l;
                                }
                            }

                            indices[j] = maxIdx;
                            gradient[i, maxIdx] = upstreamGrad[i, j];
                        }
                    }

                    return new[] { gradient };
                });

            return sorted;
        }
    }
}
