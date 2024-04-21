// ------------------------------------------------------------------------------
// <copyright file="MazeLayer.cs" author="ameritusweb" date="4/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample.VGruNetwork
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// The maze layers.
    /// </summary>
    public class MazeLayer
    {
        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] UCWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] UpdateWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] ResetWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] CandidateWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] HiddenWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] WUpdateWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] UUpdateWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] WUpdateVectors { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] UUpdateVectors { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] ZKeys { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] ZKB { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] WResetWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] UResetWeights { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] WResetVectors { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] UResetVectors { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] RKeys { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] RKB { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] IKeys { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] IKB { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] CHKeys { get; set; }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Matrix[] CHKB { get; set; }

        /// <summary>
        /// An indexer for the weights.
        /// </summary>
        /// <param name="identifier">The identifier.</param>
        /// <returns>The matrix array.</returns>
        /// <exception cref="KeyNotFoundException">Not found.</exception>
        public Matrix[] this[string identifier] => identifier switch
        {
            "UCWeights" => this.UCWeights,
            "UpdateWeights" => this.UpdateWeights,
            "ResetWeights" => this.ResetWeights,
            "CandidateWeights" => this.CandidateWeights,
            "HiddenWeights" => this.HiddenWeights,
            "WUpdateWeights" => this.WUpdateWeights,
            "UUpdateWeights" => this.UUpdateWeights,
            "WUpdateVectors" => this.WUpdateVectors,
            "UUpdateVectors" => this.UUpdateVectors,
            "ZKeys" => this.ZKeys,
            "ZKB" => this.ZKB,
            "WResetWeights" => this.WResetWeights,
            "UResetWeights" => this.UResetWeights,
            "WResetVectors" => this.WResetVectors,
            "UResetVectors" => this.UResetVectors,
            "RKeys" => this.RKeys,
            "RKB" => this.RKB,
            "IKeys" => this.IKeys,
            "IKB" => this.IKB,
            "CHKeys" => this.CHKeys,
            "CHKB" => this.CHKB,
            _ => throw new KeyNotFoundException(),
        };
    }
}
