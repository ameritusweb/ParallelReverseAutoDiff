//------------------------------------------------------------------------------
// <copyright file="MultiLayerLSTMParameters.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.LstmExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Multi-layer LSTM parameters.
    /// </summary>
    [Serializable]
    public class MultiLayerLSTMParameters
    {
        /// <summary>
        /// Gets or sets the weight matrix for the input gate.
        /// </summary>
        public DeepMatrix Wi { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the forget gate.
        /// </summary>
        public DeepMatrix Wf { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the memory cell gate.
        /// </summary>
        public DeepMatrix Wc { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output gate.
        /// </summary>
        public DeepMatrix Wo { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the input gate.
        /// </summary>
        public DeepMatrix Ui { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the forget gate.
        /// </summary>
        public DeepMatrix Uf { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the memory cell gate.
        /// </summary>
        public DeepMatrix Uc { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output gate.
        /// </summary>
        public DeepMatrix Uo { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the input gate.
        /// </summary>
        public DeepMatrix Bi { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the forget gate.
        /// </summary>
        public DeepMatrix Bf { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the memory cell gate.
        /// </summary>
        public DeepMatrix Bc { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the output gate.
        /// </summary>
        public DeepMatrix Bo { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the embedding layer.
        /// </summary>
        public Matrix Be { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the embedding layer.
        /// </summary>
        public Matrix We { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output layer.
        /// </summary>
        public Matrix V { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the output layer.
        /// </summary>
        public Matrix B { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the query layer.
        /// </summary>
        public DeepMatrix Wq { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the key layer.
        /// </summary>
        public DeepMatrix Wk { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the value layer.
        /// </summary>
        public DeepMatrix Wv { get; set; }
    }
}
