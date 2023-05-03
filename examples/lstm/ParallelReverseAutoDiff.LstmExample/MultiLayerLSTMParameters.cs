//------------------------------------------------------------------------------
// <copyright file="MultiLayerLSTMParameters.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.LstmExample
{
    /// <summary>
    /// Multi-layer LSTM parameters.
    /// </summary>
    [Serializable]
    public class MultiLayerLSTMParameters
    {
        /// <summary>
        /// Gets or sets the weight matrix for the input gate.
        /// </summary>
        public double[][][] Wi { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the forget gate.
        /// </summary>
        public double[][][] Wf { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the memory cell gate.
        /// </summary>
        public double[][][] Wc { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output gate.
        /// </summary>
        public double[][][] Wo { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the input gate.
        /// </summary>
        public double[][][] Ui { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the forget gate.
        /// </summary>
        public double[][][] Uf { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the memory cell gate.
        /// </summary>
        public double[][][] Uc { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output gate.
        /// </summary>
        public double[][][] Uo { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the input gate.
        /// </summary>
        public double[][][] Bi { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the forget gate.
        /// </summary>
        public double[][][] Bf { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the memory cell gate.
        /// </summary>
        public double[][][] Bc { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the output gate.
        /// </summary>
        public double[][][] Bo { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the embedding layer.
        /// </summary>
        public double[][] Be { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the embedding layer.
        /// </summary>
        public double[][] We { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the output layer.
        /// </summary>
        public double[][] V { get; set; }

        /// <summary>
        /// Gets or sets the bias matrix for the output layer.
        /// </summary>
        public double[][] B { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the query layer.
        /// </summary>
        public double[][][] Wq { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the key layer.
        /// </summary>
        public double[][][] Wk { get; set; }

        /// <summary>
        /// Gets or sets the weight matrix for the value layer.
        /// </summary>
        public double[][][] Wv { get; set; }
    }
}
