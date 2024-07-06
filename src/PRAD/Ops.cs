//------------------------------------------------------------------------------
// <copyright file="Ops.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD
{
    using System;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Operation types.
    /// </summary>
    public class Ops
    {
        /// <summary>
        /// Gets the add Gaussian noise op type.
        /// </summary>
        public static Type AddGaussianNoiseOpType => typeof(AddGaussianNoiseOperation);

        /// <summary>
        /// Gets the Amplified sigmoid op type.
        /// </summary>
        public static Type AmplifiedSigmoidOpType => typeof(AmplifiedSigmoidOperation);

        /// <summary>
        /// Gets the Apply dropout op type.
        /// </summary>
        public static Type ApplyDropoutOpType => typeof(ApplyDropoutOperation);

        /// <summary>
        /// Gets the Batch matrix row concatenate op type.
        /// </summary>
        public static Type BatchMatrixRowConcatenateOpType => typeof(BatchMatrixRowConcatenateOperation);

        /// <summary>
        /// Gets the Batch matrix sum op type.
        /// </summary>
        public static Type BatchMatrixSumOpType => typeof(BatchMatrixSumOperation);

        /// <summary>
        /// Gets the Batch matrix transpose op type.
        /// </summary>
        public static Type BatchMatrixTransposeOpType => typeof(BatchMatrixTransposeOperation);

        /// <summary>
        /// Gets the Batch matrix vector concatenate op type.
        /// </summary>
        public static Type BatchMatrixVectorConcatenateOpType => typeof(BatchMatrixVectorConcatenateOperation);

        /// <summary>
        /// Gets the Batch matrix vertical concatenate op type.
        /// </summary>
        public static Type BatchMatrixVerticalConcatenateOpType => typeof(BatchMatrixVerticalConcatenateOperation);

        /// <summary>
        /// Gets the Batch normalization op type.
        /// </summary>
        public static Type BatchNormalizationOpType => typeof(BatchNormalizationOperation);

        /// <summary>
        /// Gets the batch padding mask op type.
        /// </summary>
        public static Type BatchPaddingMaskOpType => typeof(BatchPaddingMaskOperation);

        /// <summary>
        /// Gets the Batch scale and shift op type.
        /// </summary>
        public static Type BatchScaleAndShiftOpType => typeof(BatchScaleAndShiftOperation);

        /// <summary>
        /// Gets the Batch sine softmax op type.
        /// </summary>
        public static Type BatchSineSoftmaxOpType => typeof(BatchSineSoftmaxOperation);

        /// <summary>
        /// Gets the Batch softmax op type.
        /// </summary>
        public static Type BatchSoftmaxOpType => typeof(SoftmaxOperation);

        /// <summary>
        /// Gets the Batch swigLU op type.
        /// </summary>
        public static Type BatchSwigLUOpType => typeof(BatchSwigLUOperation);

        /// <summary>
        /// Gets the batch swish op type.
        /// </summary>
        public static Type BatchSwishOpType => typeof(BatchSwishOperation);

        /// <summary>
        /// Gets the Batch tanh op type.
        /// </summary>
        public static Type BatchTanhOpType => typeof(BatchTanhOperation);

        /// <summary>
        /// Gets the Cosine projection op type.
        /// </summary>
        public static Type CosineProjectionOpType => typeof(CosineProjectionOperation);

        /// <summary>
        /// Gets the Cosine scaling op type.
        /// </summary>
        public static Type CosineScalingOpType => typeof(CosineScalingOperation);

        /// <summary>
        /// Gets the Deep batch normalization op type.
        /// </summary>
        public static Type DeepBatchNormalizationOpType => typeof(DeepBatchNormalizationOperation);

        /// <summary>
        /// Gets the Deep concatenate op type.
        /// </summary>
        public static Type DeepConcatenateOpType => typeof(DeepConcatenateOperation);

        /// <summary>
        /// Gets the Deep convolution op type.
        /// </summary>
        public static Type DeepConvolutionOpType => typeof(DeepConvolutionOperation);

        /// <summary>
        /// Gets the Deep matrix leaky ReLU op type.
        /// </summary>
        public static Type DeepLeakyReLUOpType => typeof(DeepLeakyReLUOperation);

        /// <summary>
        /// Gets the Deep matrix elementwise add op type.
        /// </summary>
        public static Type DeepMatrixElementwiseAddOpType => typeof(DeepMatrixElementwiseAddOperation);

        /// <summary>
        /// Gets the Deep matrix elementwise multiply sum op type.
        /// </summary>
        public static Type DeepMatrixElementwiseMultiplySumOpType => typeof(DeepMatrixElementwiseMultiplySumOperation);

        /// <summary>
        /// Gets the Deep max pool op type.
        /// </summary>
        public static Type DeepMaxPoolOpType => typeof(DeepMaxPoolOperation);

        /// <summary>
        /// Gets the Deep pairwise attention op type.
        /// </summary>
        public static Type DeepPairwiseAttentionOpType => typeof(DeepPairwiseAttentionOperation);

        /// <summary>
        /// Gets the Deep ReLU op type.
        /// </summary>
        public static Type DeepReLUOpType => typeof(DeepReLUOperation);

        /// <summary>
        /// Gets the Deep scale and shift op type.
        /// </summary>
        public static Type DeepScaleAndShiftOpType => typeof(DeepScaleAndShiftOperation);

        /// <summary>
        /// Gets the Dual weighted op type.
        /// </summary>
        public static Type DualWeightedOpType => typeof(DualWeightedOperation);

        /// <summary>
        /// Gets the Elementwise multiply and sum op type.
        /// </summary>
        public static Type ElementwiseMultiplyAndSumOpType => typeof(ElementwiseMultiplyAndSumOperation);

        /// <summary>
        /// Gets the Elementwise square and sum op type.
        /// </summary>
        public static Type ElementwiseSquareAndSumOpType => typeof(ElementwiseSquareAndSumOperation);

        /// <summary>
        /// Gets the Elementwise square op type.
        /// </summary>
        public static Type ElementwiseSquareOpType => typeof(ElementwiseSquareOperation);

        /// <summary>
        /// Gets the Elementwise vector add op type.
        /// </summary>
        public static Type ElementwiseVectorAddOpType => typeof(ElementwiseVectorAddOperation);

        /// <summary>
        /// Gets the Elementwise vector cartesian summation op type.
        /// </summary>
        public static Type ElementwiseVectorCartesianSummationOpType => typeof(ElementwiseVectorCartesianSummationOperation);

        /// <summary>
        /// Gets the Elementwise vector constituent multiply op type.
        /// </summary>
        public static Type ElementwiseVectorConstituentMultiplyOpType => typeof(ElementwiseVectorConstituentMultiplyOperation);

        /// <summary>
        /// Gets the Elementwise vector decomposition op type.
        /// </summary>
        public static Type ElementwiseVectorDecompositionOpType => typeof(ElementwiseVectorDecompositionOperation);

        /// <summary>
        /// Gets the Elementwise vector mini decomposition op type.
        /// </summary>
        public static Type ElementwiseVectorMiniDecompositionOpType => typeof(ElementwiseVectorMiniDecompositionOperation);

        /// <summary>
        /// Gets the embedding op type.
        /// </summary>
        public static Type EmbeddingOpType => typeof(EmbeddingOperation);

        /// <summary>
        /// Gets the Feature aggregation op type.
        /// </summary>
        public static Type FeatureAggregationOpType => typeof(FeatureAggregationOperation);

        /// <summary>
        /// Gets the flatten op type.
        /// </summary>
        public static Type FlattenOpType => typeof(FlattenOperation);

        /// <summary>
        /// Gets the GELU op type.
        /// </summary>
        public static Type GELUOpType => typeof(GELUOperation);

        /// <summary>
        /// Gets the GPU matrix multiply and sum op type.
        /// </summary>
        public static Type GpuMatrixMultiplyAndSumOpType => typeof(GpuMatrixMultiplyAndSumOperation);

        /// <summary>
        /// Gets the GPU matrix multiply op type.
        /// </summary>
        public static Type GpuMatrixMultiplyOpType => typeof(GpuMatrixMultiplyOperation);

        /// <summary>
        /// Gets the graph attention op type.
        /// </summary>
        public static Type GraphAttentionOpType => typeof(GraphAttentionOperation);

        /// <summary>
        /// Gets the Gravitational influence on weights op type.
        /// </summary>
        public static Type GravitationalInfluenceOnWeightsOpType => typeof(GravitationalInfluenceOnWeightsOperation);

        /// <summary>
        /// Gets the Gravitational influence op type.
        /// </summary>
        public static Type GravitationalInfluenceOpType => typeof(GravitationalInfluenceOperation);

        /// <summary>
        /// Gets the Hadamard product op type.
        /// </summary>
        public static Type HadamardProductOpType => typeof(HadamardProductOperation);

        /// <summary>
        /// Gets the Hadamard scaled product op type.
        /// </summary>
        public static Type HadamardScaledProductOpType => typeof(HadamardScaledProductOperation);

        /// <summary>
        /// Gets the Hierarchical scaling op type.
        /// </summary>
        public static Type HierarchicalScalingOpType => typeof(HierarchicalScalingOperation);

        /// <summary>
        /// Gets the Layer normalization op type.
        /// </summary>
        public static Type LayerNormalizationOpType => typeof(LayerNormalizationOperation);

        /// <summary>
        /// Gets the Leaky ReLU op type.
        /// </summary>
        public static Type LeakyReLUOpType => typeof(LeakyReLUOperation);

        /// <summary>
        /// Gets the Matrix add broadcasting op type.
        /// </summary>
        public static Type MatrixAddBroadcastingOpType => typeof(MatrixAddBroadcastingOperation);

        /// <summary>
        /// Gets the Matrix add op type.
        /// </summary>
        public static Type MatrixAddOpType => typeof(MatrixAddOperation);

        /// <summary>
        /// Gets the Matrix add scalar op type.
        /// </summary>
        public static Type MatrixAddScalarOpType => typeof(MatrixAddScalarOperation);

        /// <summary>
        /// Gets the Matrix add three op type.
        /// </summary>
        public static Type MatrixAddThreeOpType => typeof(MatrixAddThreeOperation);

        /// <summary>
        /// Gets the Matrix average op type.
        /// </summary>
        public static Type MatrixAverageOpType => typeof(MatrixAverageOperation);

        /// <summary>
        /// Gets the Matrix broadcast op type.
        /// </summary>
        public static Type MatrixBroadcastOpType => typeof(MatrixBroadcastOperation);

        /// <summary>
        /// Gets the Matrix concatenate op type.
        /// </summary>
        public static Type MatrixConcatenateOpType => typeof(MatrixConcatenateOperation);

        /// <summary>
        /// Gets the Matrix diagonal filter op type.
        /// </summary>
        public static Type MatrixDiagonalFilterOpType => typeof(MatrixDiagonalFilterOperation);

        /// <summary>
        /// Gets the Matrix horizontal concatenate op type.
        /// </summary>
        public static Type MatrixHorizontalConcatenateOpType => typeof(MatrixHorizontalConcatenateOperation);

        /// <summary>
        /// Gets the Matrix multiply and sum op type.
        /// </summary>
        public static Type MatrixMultiplyAndSumOpType => typeof(MatrixMultiplyAndSumOperation);

        /// <summary>
        /// Gets the Matrix multiply and sum rows op type.
        /// </summary>
        public static Type MatrixMultiplyAndSumRowsOpType => typeof(MatrixMultiplyAndSumRowsOperation);

        /// <summary>
        /// Gets the matrix multiply op type.
        /// </summary>
        public static Type MatrixMultiplyOpType => typeof(MatrixMultiplyOperation);

        /// <summary>
        /// Gets the Matrix multiply scalar op type.
        /// </summary>
        public static Type MatrixMultiplyScalarOpType => typeof(MatrixMultiplyScalarOperation);

        /// <summary>
        /// Gets the matrix row concatenate op type.
        /// </summary>
        public static Type MatrixRowConcatenateOpType => typeof(MatrixRowConcatenateOperation);

        /// <summary>
        /// Gets the Matrix sum op type.
        /// </summary>
        public static Type MatrixSumOpType => typeof(MatrixSumOperation);

        /// <summary>
        /// Gets the Matrix transpose op type.
        /// </summary>
        public static Type MatrixTransposeOpType => typeof(MatrixTransposeOperation);

        /// <summary>
        /// Gets the Matrix vector concatenate op type.
        /// </summary>
        public static Type MatrixVectorConcatenateOpType => typeof(MatrixVectorConcatenateOperation);

        /// <summary>
        /// Gets the Matrix vertical concatenate op type.
        /// </summary>
        public static Type MatrixVerticalConcatenateOpType => typeof(MatrixVerticalConcatenateOperation);

        /// <summary>
        /// Gets the Modified softmax op type.
        /// </summary>
        public static Type ModifiedSoftmaxOpType => typeof(ModifiedSoftmaxOperation);

        /// <summary>
        /// Gets the Multi-query self-attention op type.
        /// </summary>
        public static Type MultiQuerySelfAttentionOpType => typeof(MultiQuerySelfAttentionOperation);

        /// <summary>
        /// Gets the Multi-row modified softmax op type.
        /// </summary>
        public static Type MultiRowModifiedSoftmaxOpType => typeof(MultiRowModifiedSoftmaxOperation);

        /// <summary>
        /// Gets the padding mask op type.
        /// </summary>
        public static Type PaddingMaskOpType => typeof(PaddingMaskOperation);

        /// <summary>
        /// Gets the Pairwise sine softmax op type.
        /// </summary>
        public static Type PairwiseSineSoftmaxOpType => typeof(PairwiseSineSoftmaxOperation);

        /// <summary>
        /// Gets the piece-wise activation op type.
        /// </summary>
        public static Type PiecewiseActivationOperation => typeof(PiecewiseActivationOperation);

        /// <summary>
        /// Gets the ReLU op type.
        /// </summary>
        public static Type ReLUOpType => typeof(ReLUOperation);

        /// <summary>
        /// Gets the RMSNorm op type.
        /// </summary>
        public static Type RMSNormOpType => typeof(RMSNormOperation);

        /// <summary>
        /// Gets the Scale and shift op type.
        /// </summary>
        public static Type ScaleAndShiftOpType => typeof(ScaleAndShiftOperation);

        /// <summary>
        /// Gets the Sigmoid op type.
        /// </summary>
        public static Type SigmoidOpType => typeof(SigmoidOperation);

        /// <summary>
        /// Gets the Sigmoid shift op type.
        /// </summary>
        public static Type SigmoidShiftOpType => typeof(SigmoidShiftOperation);

        /// <summary>
        /// Gets the Sine softmax op type.
        /// </summary>
        public static Type SineSoftmaxOpType => typeof(SineSoftmaxOperation);

        /// <summary>
        /// Gets the Softmax op type.
        /// </summary>
        public static Type SoftmaxOpType => typeof(SoftmaxOperation);

        /// <summary>
        /// Gets the Stretched sigmoid op type.
        /// </summary>
        public static Type StretchedSigmoidOpType => typeof(StretchedSigmoidOperation);

        /// <summary>
        /// Gets the SwigLU op type.
        /// </summary>
        public static Type SwigLUOpType => typeof(SwigLUOperation);

        /// <summary>
        /// Gets the Swish op type.
        /// </summary>
        public static Type SwishOpType => typeof(SwishOperation);

        /// <summary>
        /// Gets the Take left op type.
        /// </summary>
        public static Type TakeLeftOpType => typeof(TakeLeftOperation);

        /// <summary>
        /// Gets the Take right op type.
        /// </summary>
        public static Type TakeRightOpType => typeof(TakeRightOperation);

        /// <summary>
        /// Gets the Tanh op type.
        /// </summary>
        public static Type TanhOpType => typeof(TanhOperation);

        /// <summary>
        /// Gets the Varied masked iterative softmax op type.
        /// </summary>
        public static Type VariedMaskedIterativeSoftmaxOpType => typeof(VariedMaskedIterativeSoftmaxOperation);

        /// <summary>
        /// Gets the Varied masked softmax op type.
        /// </summary>
        public static Type VariedMaskedSoftmaxOpType => typeof(VariedMaskedSoftmaxOperation);

        /// <summary>
        /// Gets the Varied softmax op type.
        /// </summary>
        public static Type VariedSoftmaxOpType => typeof(VariedSoftmaxOperation);

        /// <summary>
        /// Gets the Vector attention binary op type.
        /// </summary>
        public static Type VectorAttentionBinaryOpType => typeof(VectorAttentionBinaryOperation);

        /// <summary>
        /// Gets the Vector attention op type.
        /// </summary>
        public static Type VectorAttentionOpType => typeof(VectorAttentionOperation);

        /// <summary>
        /// Gets the Vectorize op type.
        /// </summary>
        public static Type VectorizeOpType => typeof(VectorizeOperation);
    }
}
