﻿using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.RMAD;

namespace SwinExample.RMAD
{
    /// <summary>
    /// Operation that reverses window partitioning to restore original feature map dimensions.
    /// </summary>
    public class WindowReverseOperation : Operation
    {
        private PradOp input;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new WindowReverseOperation();
        }

        /// <summary>
        /// Performs the forward operation for reversing window partitioning.
        /// </summary>
        /// <param name="input">Input tensor of shape [B*num_windows, window_size*window_size, C]</param>
        /// <param name="options">Options tensor containing [windowSize, height, width]</param>
        /// <returns>Output tensor of shape [B, H, W, C]</returns>
        public Matrix Forward(Matrix input, Matrix options)
        {
            SwinTransformerTools tools = new SwinTransformerTools();
            
            // Convert inputs to PradOp
            this.input = new PradOp(input.ToTensor());
            var optionsPradOp = new PradOp(options.ToTensor());

            // Reverse window partitioning
            var result = tools.WindowReverse(this.input, optionsPradOp);
            
            this.resultOp = result.PradOp;
            this.Output = result.Result.ToMatrix();
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            // Perform backward pass using PradOp
            this.resultOp.Back(dLdOutput.ToTensor());

            // Get gradient with respect to input
            var dInput = this.input.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dInput)
                .Build();
        }
    }
}