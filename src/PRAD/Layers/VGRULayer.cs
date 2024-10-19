//------------------------------------------------------------------------------
// <copyright file="VGRULayer.cs" author="ameritusweb" date="6/20/2024">
// Copyright (c) 2024 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.PRAD.Layers
{
    /// <summary>
    /// A Vectorized Gated Recurrent Unit (VGRU) layer.
    /// This layer performs operations similar to GRU but using vector-based neural network structures,
    /// handling magnitude and angle representations with attention mechanisms and custom vector-based computations.
    /// </summary>
    public class VGRULayer
    {
        private PradVectorTools vectorTools = new PradVectorTools();
        private PradOp opInput;
        private PradOp opAngles;
        private PradOp decompositionWeights;
        private PradOp decompositionVectors;
        private PradOp previousHiddenState;
        private PradOp[][] updateWeights;
        private PradOp[][] resetWeights;
        private PradOp[][] candidateWeights;
        private PradOp[][] hiddenWeights;
        private PradOp convolutionFilter;

        /// <summary>
        /// Initializes a new instance of the <see cref="VGRULayer"/> class.
        /// </summary>
        /// <param name="opInput">The input tensor for the current time step.</param>
        /// <param name="opAngles">The angle tensor corresponding to the input tensor.</param>
        /// <param name="decompositionVectors">The vectors used in vector decomposition.</param>
        /// <param name="decompositionWeights">The weights used in vector decomposition.</param>
        /// <param name="previousHiddenState">The hidden state tensor from the previous time step.</param>
        /// <param name="updateWeights">Weights for the update gate in each layer.</param>
        /// <param name="resetWeights">Weights for the reset gate in each layer.</param>
        /// <param name="candidateWeights">Weights for the candidate hidden state computations.</param>
        /// <param name="hiddenWeights">Weights for updating the hidden state.</param>
        /// <param name="convolutionFilter">The filter used for custom convolution on the hidden state.</param>
        public VGRULayer(
            PradOp opInput,
            PradOp opAngles,
            PradOp decompositionVectors,
            PradOp decompositionWeights,
            PradOp previousHiddenState,
            PradOp[][] updateWeights,
            PradOp[][] resetWeights,
            PradOp[][] candidateWeights,
            PradOp[][] hiddenWeights,
            PradOp convolutionFilter)
        {
            this.opInput = opInput;
            this.opAngles = opAngles;
            this.decompositionVectors = decompositionVectors;
            this.decompositionWeights = decompositionWeights;
            this.previousHiddenState = previousHiddenState;
            this.updateWeights = updateWeights;
            this.resetWeights = resetWeights;
            this.candidateWeights = candidateWeights;
            this.hiddenWeights = hiddenWeights;
            this.convolutionFilter = convolutionFilter;
        }

        /// <summary>
        /// Computes the output of the VGRU layer for a given time step.
        /// <br />
        /// This function performs the following steps:
        /// - Vectorizes the input and angles.
        /// - Decomposes the vectors based on the decomposition weights and vectors.
        /// - Computes the update (z), reset (r), and candidate hidden states using custom vector operations.
        /// - Updates the hidden state based on a combination of the previous hidden state and candidate hidden state.
        /// - Performs a custom vector convolution to produce the final output.
        /// </summary>
        /// <returns>The final output tensor of the VGRU layer after computation.</returns>
        public PradResult Compute()
        {
            var squaredWeights = this.vectorTools.ElementwiseSquare(this.decompositionWeights);
            var squaredAngles = this.vectorTools.ElementwiseSquare(this.opAngles);
            var vectorizedInput = this.vectorTools.Vectorize(this.opInput, squaredAngles.PradOp);
            var decomposed = this.vectorTools.VectorDecomposition(vectorizedInput.PradOp, this.decompositionVectors, squaredWeights.PradOp);

            PradOp hiddenState = this.previousHiddenState;
            PradResult currentInput = decomposed;

            // Layer Operations
            for (int layer = 0; layer < this.updateWeights.Length; layer++)
            {
                // Update gate (z) computation
                var currentInputBranches = currentInput.BranchStack(2);
                var hiddenStateBranches = hiddenState.BranchStack(3);
                var z = this.ComputeGate(currentInput.PradOp, hiddenStateBranches.Pop(), this.updateWeights[layer]);

                // Reset gate (r) computation
                var r = this.ComputeGate(currentInputBranches.Pop(), hiddenStateBranches.Pop(), this.resetWeights[layer]);

                // Candidate hidden state computation
                var candidateHidden = this.ComputeCandidateHiddenState(currentInputBranches.Pop(), hiddenStateBranches.Pop(), r.PradOp, this.candidateWeights[layer]);

                // New hidden state computation
                var newHiddenState = this.UpdateHiddenState(hiddenState, z.PradOp, candidateHidden.PradOp, this.hiddenWeights[layer][0]);
                hiddenState = newHiddenState.PradOp;

                // Update currentInput for the next layer
                currentInput = newHiddenState;
            }

            // End Operations: Convolution on the final hidden state
            var splitHiddenState = this.vectorTools.SplitInterleavedTensor(hiddenState);
            var splitFilter = this.vectorTools.SplitInterleavedTensor(this.convolutionFilter);
            var output = this.vectorTools.CustomVectorConvolution(splitHiddenState.Item1.PradOp, splitHiddenState.Item2.PradOp, splitFilter.Item1.PradOp, splitFilter.Item2.PradOp);

            return output;
        }

        /// <summary>
        /// Computes the gate operation (update or reset) for the VGRU layer.
        /// This includes vector matrix multiplication, weighted addition, activation, and attention.
        /// </summary>
        /// <param name="currentInput">The current input tensor.</param>
        /// <param name="previousHiddenState">The previous hidden state tensor.</param>
        /// <param name="weights">The weights for the gate (update or reset).</param>
        /// <returns>The resulting gate activation.</returns>
        private PradResult ComputeGate(PradOp currentInput, PradOp previousHiddenState, PradOp[] weights)
        {
            var currentWeights = this.vectorTools.ElementwiseSquare(weights[0]);
            var previousWeights = this.vectorTools.ElementwiseSquare(weights[1]);

            var currentGate = this.vectorTools.VectorBasedMatrixMultiplication(currentInput, weights[2], currentWeights.PradOp);
            var previousGate = this.vectorTools.VectorBasedMatrixMultiplication(previousHiddenState, weights[3], previousWeights.PradOp);

            var combinedGate = this.vectorTools.VectorWeightedAdd(currentGate.PradOp, previousGate.PradOp, weights[4]);
            var combinedGateBranch = combinedGate.Branch();
            var gateKeys = this.vectorTools.MatrixMultiplication(combinedGateBranch, weights[5]);
            var gateKeysBroadcasted = this.vectorTools.AddBroadcasting(gateKeys.PradOp, weights[6]);

            var activatedGate = this.vectorTools.LeakyReLU(gateKeysBroadcasted.PradOp);
            var softmaxGate = this.vectorTools.PairwiseSineSoftmax(activatedGate.PradOp);

            return this.vectorTools.VectorAttention(combinedGate.PradOp, softmaxGate.PradOp);
        }

        /// <summary>
        /// Computes the candidate hidden state for the VGRU layer.
        /// </summary>
        /// <param name="currentInput">The current input tensor.</param>
        /// <param name="hiddenState">The previous hidden state tensor.</param>
        /// <param name="resetGate">The result of the reset gate operation.</param>
        /// <param name="weights">The weights for the candidate hidden state computation.</param>
        /// <returns>The candidate hidden state result.</returns>
        private PradResult ComputeCandidateHiddenState(PradOp currentInput, PradOp hiddenState, PradOp resetGate, PradOp[] weights)
        {
            var currentInputBranch = currentInput.Branch();
            var weightedHiddenState = this.vectorTools.VectorWeightedAdd(hiddenState, resetGate, weights[0]);
            var inputKeys = this.vectorTools.MatrixMultiplication(currentInputBranch, weights[1]);
            var inputKeysBroadcasted = this.vectorTools.AddBroadcasting(inputKeys.PradOp, weights[2]);

            var activatedInput = this.vectorTools.LeakyReLU(inputKeysBroadcasted.PradOp);
            var softmaxInput = this.vectorTools.PairwiseSineSoftmax(activatedInput.PradOp);

            var inputAttention = this.vectorTools.VectorAttention(currentInput, softmaxInput.PradOp);
            return this.vectorTools.VectorWeightedAdd(inputAttention.PradOp, weightedHiddenState.PradOp, weights[3]);
        }

        /// <summary>
        /// Updates the hidden state by combining the previous hidden state with the candidate hidden state.
        /// </summary>
        /// <param name="hiddenState">The previous hidden state tensor.</param>
        /// <param name="updateGate">The update gate result.</param>
        /// <param name="candidateHiddenState">The candidate hidden state result.</param>
        /// <param name="hiddenWeight">The weight used for updating the hidden state.</param>
        /// <returns>The new hidden state result.</returns>
        private PradResult UpdateHiddenState(PradOp hiddenState, PradOp updateGate, PradOp candidateHiddenState, PradOp hiddenWeight)
        {
            var updateGateBranch = updateGate.Branch();
            var complementUpdateGate = this.vectorTools.ElementwiseInversion(updateGate);

            var weightedPreviousHiddenState = this.vectorTools.VectorAveraging(hiddenState, updateGateBranch);
            var weightedCandidateHiddenState = this.vectorTools.VectorAveraging(candidateHiddenState, complementUpdateGate.PradOp);

            return this.vectorTools.VectorWeightedAdd(weightedPreviousHiddenState.PradOp, weightedCandidateHiddenState.PradOp, hiddenWeight);
        }
    }
}
