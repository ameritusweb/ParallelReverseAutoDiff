//------------------------------------------------------------------------------
// <copyright file="PradOp.Ops.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The ops for the PradOp class.
    /// </summary>
    public partial class PradOp
    {
        /// <summary>
        /// Operation types.
        /// </summary>
        public class Ops
        {
            /// <summary>
            /// Gets the add Gaussian noise op.
            /// </summary>
            public static Func<double, AddGaussianNoiseOperation> AddGaussianNoiseOp => (d) => new AddGaussianNoiseOperation(d);

            /// <summary>
            /// Gets the Amplified sigmoid op.
            /// </summary>
            public static Func<AmplifiedSigmoidOperation> AmplifiedSigmoidOp => () => new AmplifiedSigmoidOperation();

            /// <summary>
            /// Gets the Apply dropout op.
            /// </summary>
            public static Func<double, ApplyDropoutOperation> ApplyDropoutOp => (d) => new ApplyDropoutOperation(d);

            /// <summary>
            /// Gets the Cosine projection op.
            /// </summary>
            public static Func<CosineProjectionOperation> CosineProjectionOp => () => new CosineProjectionOperation();

            /// <summary>
            /// Gets the Cosine scaling op.
            /// </summary>
            public static Func<CosineScalingOperation> CosineScalingOp => () => new CosineScalingOperation();

            /// <summary>
            /// Gets the Deep batch normalization op.
            /// </summary>
            public static Func<DeepBatchNormalizationOperation> DeepBatchNormalizationOp => () => new DeepBatchNormalizationOperation();

            /// <summary>
            /// Gets the Deep concatenate op.
            /// </summary>
            public static Func<DeepConcatenateOperation> DeepConcatenateOp => () => new DeepConcatenateOperation();

            /// <summary>
            /// Gets the Deep convolution op.
            /// </summary>
            public static Func<int, DeepConvolutionOperation> DeepConvolutionOp => (i) => new DeepConvolutionOperation(i);

            /// <summary>
            /// Gets the Deep matrix leaky ReLU op.
            /// </summary>
            public static Func<double, DeepLeakyReLUOperation> DeepLeakyReLUOp => (d) => new DeepLeakyReLUOperation(d);

            /// <summary>
            /// Gets the Deep matrix elementwise add op.
            /// </summary>
            public static Func<DeepMatrixElementwiseAddOperation> DeepMatrixElementwiseAddOp => () => new DeepMatrixElementwiseAddOperation();

            /// <summary>
            /// Gets the Deep matrix elementwise multiply sum op.
            /// </summary>
            public static Func<DeepMatrixElementwiseMultiplySumOperation> DeepMatrixElementwiseMultiplySumOp => () => new DeepMatrixElementwiseMultiplySumOperation();

            /// <summary>
            /// Gets the Deep max pool op.
            /// </summary>
            public static Func<int, DeepMaxPoolOperation> DeepMaxPoolOp => (i) => new DeepMaxPoolOperation(i);

            /// <summary>
            /// Gets the Deep pairwise attention op.
            /// </summary>
            public static Func<DeepPairwiseAttentionOperation> DeepPairwiseAttentionOp => () => new DeepPairwiseAttentionOperation();

            /// <summary>
            /// Gets the Deep ReLU op.
            /// </summary>
            public static Func<DeepReLUOperation> DeepReLUOp => () => new DeepReLUOperation();

            /// <summary>
            /// Gets the Deep scale and shift op.
            /// </summary>
            public static Func<DeepScaleAndShiftOperation> DeepScaleAndShiftOp => () => new DeepScaleAndShiftOperation();

            /// <summary>
            /// Gets the Dual weighted op.
            /// </summary>
            public static Func<DualWeightedOperation> DualWeightedOp => () => new DualWeightedOperation();

            /// <summary>
            /// Gets the Elementwise multiply and sum op.
            /// </summary>
            public static Func<ElementwiseMultiplyAndSumOperation> ElementwiseMultiplyAndSumOp => () => new ElementwiseMultiplyAndSumOperation();

            /// <summary>
            /// Gets the Elementwise square and sum op.
            /// </summary>
            public static Func<ElementwiseSquareAndSumOperation> ElementwiseSquareAndSumOp => () => new ElementwiseSquareAndSumOperation();

            /// <summary>
            /// Gets the Elementwise square op.
            /// </summary>
            public static Func<ElementwiseSquareOperation> ElementwiseSquareOp => () => new ElementwiseSquareOperation();

            /// <summary>
            /// Gets the Elementwise vector add op.
            /// </summary>
            public static Func<ElementwiseVectorAddOperation> ElementwiseVectorAddOp => () => new ElementwiseVectorAddOperation();

            /// <summary>
            /// Gets the Elementwise vector cartesian summation op.
            /// </summary>
            public static Func<ElementwiseVectorCartesianSummationOperation> ElementwiseVectorCartesianSummationOp => () => new ElementwiseVectorCartesianSummationOperation();

            /// <summary>
            /// Gets the Elementwise vector constituent multiply op.
            /// </summary>
            public static Func<ElementwiseVectorConstituentMultiplyOperation> ElementwiseVectorConstituentMultiplyOp => () => new ElementwiseVectorConstituentMultiplyOperation();

            /// <summary>
            /// Gets the Elementwise vector decomposition op.
            /// </summary>
            public static Func<ElementwiseVectorDecompositionOperation> ElementwiseVectorDecompositionOp => () => new ElementwiseVectorDecompositionOperation();

            /// <summary>
            /// Gets the Elementwise vector mini decomposition op.
            /// </summary>
            public static Func<ElementwiseVectorMiniDecompositionOperation> ElementwiseVectorMiniDecompositionOp => () => new ElementwiseVectorMiniDecompositionOperation();

            /// <summary>
            /// Gets the embedding op.
            /// </summary>
            public static Func<EmbeddingOperation> EmbeddingOp => () => new EmbeddingOperation();

            /// <summary>
            /// Gets the Feature aggregation op.
            /// </summary>
            public static Func<FeatureAggregationOperation> FeatureAggregationOp => () => new FeatureAggregationOperation();

            /// <summary>
            /// Gets the flatten op.
            /// </summary>
            public static Func<FlattenOperation> FlattenOp => () => new FlattenOperation();

            /// <summary>
            /// Gets the GELU op.
            /// </summary>
            public static Func<GELUOperation> GELUOp => () => new GELUOperation();

            /// <summary>
            /// Gets the GPU matrix multiply and sum op.
            /// </summary>
            public static Func<GpuMatrixMultiplyAndSumOperation> GpuMatrixMultiplyAndSumOp => () => new GpuMatrixMultiplyAndSumOperation();

            /// <summary>
            /// Gets the GPU matrix multiply op.
            /// </summary>
            public static Func<GpuMatrixMultiplyOperation> GpuMatrixMultiplyOp => () => new GpuMatrixMultiplyOperation();

            /// <summary>
            /// Gets the graph attention op.
            /// </summary>
            public static Func<GraphAttentionOperation> GraphAttentionOp => () => new GraphAttentionOperation();

            /// <summary>
            /// Gets the Gravitational influence on weights op.
            /// </summary>
            public static Func<GravitationalInfluenceOnWeightsOperation> GravitationalInfluenceOnWeightsOp => () => new GravitationalInfluenceOnWeightsOperation();

            /// <summary>
            /// Gets the Gravitational influence op.
            /// </summary>
            public static Func<GravitationalInfluenceOperation> GravitationalInfluenceOp => () => new GravitationalInfluenceOperation();

            /// <summary>
            /// Gets the Hadamard product op.
            /// </summary>
            public static Func<HadamardProductOperation> HadamardProductOp => () => new HadamardProductOperation();

            /// <summary>
            /// Gets the Hadamard scaled product op.
            /// </summary>
            public static Func<HadamardScaledProductOperation> HadamardScaledProductOp => () => new HadamardScaledProductOperation();

            /// <summary>
            /// Gets the Hierarchical scaling op.
            /// </summary>
            public static Func<HierarchicalScalingOperation> HierarchicalScalingOp => () => new HierarchicalScalingOperation();

            /// <summary>
            /// Gets the Layer normalization op.
            /// </summary>
            public static Func<LayerNormalizationOperation> LayerNormalizationOp => () => new LayerNormalizationOperation();

            /// <summary>
            /// Gets the Leaky ReLU op.
            /// </summary>
            public static Func<double, LeakyReLUOperation> LeakyReLUOp => (d) => new LeakyReLUOperation(d);

            /// <summary>
            /// Gets the Matrix add broadcasting op.
            /// </summary>
            public static Func<MatrixAddBroadcastingOperation> MatrixAddBroadcastingOp => () => new MatrixAddBroadcastingOperation();

            /// <summary>
            /// Gets the Matrix add op.
            /// </summary>
            public static Func<MatrixAddOperation> MatrixAddOp => () => new MatrixAddOperation();

            /// <summary>
            /// Gets the Matrix add scalar op.
            /// </summary>
            public static Func<MatrixAddScalarOperation> MatrixAddScalarOp => () => new MatrixAddScalarOperation();

            /// <summary>
            /// Gets the Matrix add three op.
            /// </summary>
            public static Func<MatrixAddThreeOperation> MatrixAddThreeOp => () => new MatrixAddThreeOperation();

            /// <summary>
            /// Gets the Matrix average op.
            /// </summary>
            public static Func<MatrixAverageOperation> MatrixAverageOp => () => new MatrixAverageOperation();

            /// <summary>
            /// Gets the Matrix broadcast op.
            /// </summary>
            public static Func<MatrixBroadcastOperation> MatrixBroadcastOp => () => new MatrixBroadcastOperation();

            /// <summary>
            /// Gets the Matrix concatenate op.
            /// </summary>
            public static Func<MatrixConcatenateOperation> MatrixConcatenateOp => () => new MatrixConcatenateOperation();

            /// <summary>
            /// Gets the Matrix diagonal filter op.
            /// </summary>
            public static Func<MatrixDiagonalFilterOperation> MatrixDiagonalFilterOp => () => new MatrixDiagonalFilterOperation();

            /// <summary>
            /// Gets the Matrix horizontal concatenate op.
            /// </summary>
            public static Func<MatrixHorizontalConcatenateOperation> MatrixHorizontalConcatenateOp => () => new MatrixHorizontalConcatenateOperation();

            /// <summary>
            /// Gets the Matrix multiply and sum op.
            /// </summary>
            public static Func<MatrixMultiplyAndSumOperation> MatrixMultiplyAndSumOp => () => new MatrixMultiplyAndSumOperation();

            /// <summary>
            /// Gets the Matrix multiply and sum rows op.
            /// </summary>
            public static Func<MatrixMultiplyAndSumRowsOperation> MatrixMultiplyAndSumRowsOp => () => new MatrixMultiplyAndSumRowsOperation();

            /// <summary>
            /// Gets the matrix multiply op.
            /// </summary>
            public static Func<MatrixMultiplyOperation> MatrixMultiplyOp => () => new MatrixMultiplyOperation();

            /// <summary>
            /// Gets the Matrix multiply scalar op.
            /// </summary>
            public static Func<MatrixMultiplyScalarOperation> MatrixMultiplyScalarOp => () => new MatrixMultiplyScalarOperation();

            /// <summary>
            /// Gets the matrix row concatenate op.
            /// </summary>
            public static Func<MatrixRowConcatenateOperation> MatrixRowConcatenateOp => () => new MatrixRowConcatenateOperation();

            /// <summary>
            /// Gets the Matrix sum op.
            /// </summary>
            public static Func<MatrixSumOperation> MatrixSumOp => () => new MatrixSumOperation();

            /// <summary>
            /// Gets the Matrix transpose op.
            /// </summary>
            public static Func<MatrixTransposeOperation> MatrixTransposeOp => () => new MatrixTransposeOperation();

            /// <summary>
            /// Gets the Matrix vector concatenate op.
            /// </summary>
            public static Func<MatrixVectorConcatenateOperation> MatrixVectorConcatenateOp => () => new MatrixVectorConcatenateOperation();

            /// <summary>
            /// Gets the Matrix vertical concatenate op.
            /// </summary>
            public static Func<MatrixVerticalConcatenateOperation> MatrixVerticalConcatenateOp => () => new MatrixVerticalConcatenateOperation();

            /// <summary>
            /// Gets the Modified softmax op.
            /// </summary>
            public static Func<ModifiedSoftmaxOperation> ModifiedSoftmaxOp => () => new ModifiedSoftmaxOperation();

            /// <summary>
            /// Gets the Multi-query self-attention op.
            /// </summary>
            public static Func<MultiQuerySelfAttentionOperation> MultiQuerySelfAttentionOp => () => new MultiQuerySelfAttentionOperation();

            /// <summary>
            /// Gets the Multi-row modified softmax op.
            /// </summary>
            public static Func<MultiRowModifiedSoftmaxOperation> MultiRowModifiedSoftmaxOp => () => new MultiRowModifiedSoftmaxOperation();

            /// <summary>
            /// Gets the padding mask op.
            /// </summary>
            public static Func<PaddingMaskOperation> PaddingMaskOp => () => new PaddingMaskOperation();

            /// <summary>
            /// Gets the Pairwise sine softmax op.
            /// </summary>
            public static Func<PairwiseSineSoftmaxOperation> PairwiseSineSoftmaxOp => () => new PairwiseSineSoftmaxOperation();

            /// <summary>
            /// Gets the piece-wise activation op.
            /// </summary>
            public static Func<PiecewiseActivationOperation> PiecewiseActivationOp => () => new PiecewiseActivationOperation();

            /// <summary>
            /// Gets the ReLU op.
            /// </summary>
            public static Func<ReLUOperation> ReLUOp => () => new ReLUOperation();

            /// <summary>
            /// Gets the RMSNorm op.
            /// </summary>
            public static Func<RMSNormOperation> RMSNormOp => () => new RMSNormOperation();

            /// <summary>
            /// Gets the Scale and shift op.
            /// </summary>
            public static Func<ScaleAndShiftOperation> ScaleAndShiftOp => () => new ScaleAndShiftOperation();

            /// <summary>
            /// Gets the Sigmoid op.
            /// </summary>
            public static Func<SigmoidOperation> SigmoidOp => () => new SigmoidOperation();

            /// <summary>
            /// Gets the Sigmoid shift op.
            /// </summary>
            public static Func<SigmoidShiftOperation> SigmoidShiftOp => () => new SigmoidShiftOperation();

            /// <summary>
            /// Gets the Sine softmax op.
            /// </summary>
            public static Func<SineSoftmaxOperation> SineSoftmaxOp => () => new SineSoftmaxOperation();

            /// <summary>
            /// Gets the Softmax op.
            /// </summary>
            public static Func<SoftmaxOperation> SoftmaxOp => () => new SoftmaxOperation();

            /// <summary>
            /// Gets the Stretched sigmoid op.
            /// </summary>
            public static Func<StretchedSigmoidOperation> StretchedSigmoidOp => () => new StretchedSigmoidOperation();

            /// <summary>
            /// Gets the SwigLU op.
            /// </summary>
            public static Func<double, SwigLUOperation> SwigLUOp => (d) => new SwigLUOperation(d);

            /// <summary>
            /// Gets the Swish op.
            /// </summary>
            public static Func<SwishOperation> SwishOp => () => new SwishOperation();

            /// <summary>
            /// Gets the Take left op.
            /// </summary>
            public static Func<TakeLeftOperation> TakeLeftOp => () => new TakeLeftOperation();

            /// <summary>
            /// Gets the Take right op.
            /// </summary>
            public static Func<TakeRightOperation> TakeRightOp => () => new TakeRightOperation();

            /// <summary>
            /// Gets the Tanh op.
            /// </summary>
            public static Func<TanhOperation> TanhOp => () => new TanhOperation();

            /// <summary>
            /// Gets the Varied masked iterative softmax op.
            /// </summary>
            public static Func<VariedMaskedIterativeSoftmaxOperation> VariedMaskedIterativeSoftmaxOp => () => new VariedMaskedIterativeSoftmaxOperation();

            /// <summary>
            /// Gets the Varied masked softmax op.
            /// </summary>
            public static Func<VariedMaskedSoftmaxOperation> VariedMaskedSoftmaxOp => () => new VariedMaskedSoftmaxOperation();

            /// <summary>
            /// Gets the Varied softmax op.
            /// </summary>
            public static Func<VariedSoftmaxOperation> VariedSoftmaxOp => () => new VariedSoftmaxOperation();

            /// <summary>
            /// Gets the Vector attention binary op.
            /// </summary>
            public static Func<VectorAttentionBinaryOperation> VectorAttentionBinaryOp => () => new VectorAttentionBinaryOperation();

            /// <summary>
            /// Gets the Vector attention op.
            /// </summary>
            public static Func<VectorAttentionOperation> VectorAttentionOp => () => new VectorAttentionOperation();

            /// <summary>
            /// Gets the Vectorize op.
            /// </summary>
            public static Func<VectorizeOperation> VectorizeOp => () => new VectorizeOperation();
        }
    }
}
