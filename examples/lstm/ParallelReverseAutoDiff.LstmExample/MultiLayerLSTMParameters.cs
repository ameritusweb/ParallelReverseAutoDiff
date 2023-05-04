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
        public Matrix[] Wi { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the forget gate.
        /// </summary>
        public Matrix[] Wf { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the memory cell gate.
        /// </summary>
        public Matrix[] Wc { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output gate.
        /// </summary>
        public Matrix[] Wo { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the input gate.
        /// </summary>
        public Matrix[] Ui { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the forget gate.
        /// </summary>
        public Matrix[] Uf { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the memory cell gate.
        /// </summary>
        public Matrix[] Uc { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output gate.
        /// </summary>
        public Matrix[] Uo { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the input gate.
        /// </summary>
        public Matrix[] Bi { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the forget gate.
        /// </summary>
        public Matrix[] Bf { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the memory cell gate.
        /// </summary>
        public Matrix[] Bc { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the output gate.
        /// </summary>
        public Matrix[] Bo { get; set; }

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
        public Matrix[] Wq { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the key layer.
        /// </summary>
        public Matrix[] Wk { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the value layer.
        /// </summary>
        public Matrix[] Wv { get; set; }
    }
}
