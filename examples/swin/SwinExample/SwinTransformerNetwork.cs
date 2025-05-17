using ParallelReverseAutoDiff.RMAD;

namespace SwinExample
{
    public class SwinTransformerNetwork : NeuralNetwork
    {
        private readonly SwinTransformerModel model;
        private SwinTransformerComputationGraph computationGraph;

        public SwinTransformerNetwork(int imageSize, int numClasses)
        {
            this.model = new SwinTransformerModel(this, imageSize, numClasses);
            this.InitializeState();
        }

        public Matrix Output { get; private set; }

        public async Task Initialize()
        {
            this.computationGraph = new SwinTransformerComputationGraph(this);
            await this.computationGraph.Initialize();
        }

        public void AutomaticForwardPropagate(Matrix input)
        {
            this.ClearState();

            var op = this.computationGraph.StartOperation;
            if (op == null)
            {
                throw new Exception("Start operation should not be null.");
            }

            IOperationBase? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);
                var forward = op.OperationType.GetMethod("Forward") ?? throw new Exception($"Forward method should exist on operation of type {op.OperationType.Name}.");
                forward.Invoke(op, parameters);

                if (op.ResultToName != null)
                {
                    var split = op.ResultToName.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                    var oo = this.computationGraph[MatrixType.Intermediate, split[0], op.LayerInfo];
                    op.CopyResult(oo);
                }

                currOp = op;
                if (op.HasNext)
                {
                    op = op.Next;
                }
            }
            while (currOp.Next != null);
        }

        public async Task<Matrix> AutomaticBackwardPropagate(Matrix gradient)
        {
            IOperationBase? backwardStartOperation = this.computationGraph["logits_0_0"];

            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(
                    Guid.NewGuid().ToString(),
                    backwardStartOperation,
                    0);
                await opVisitor.TraverseAsync();
                opVisitor.Reset();
            }

            return gradient;
        }

        private void InitializeState()
        {
            var output = new Matrix(CommonMatrixUtils.InitializeZeroMatrix(this.model.NumClasses, 1));

            if (this.Output == null)
            {
                this.Output = output;
            }
            else
            {
                this.Output.Replace(output.ToArray());
            }
        }

        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] {
            this.model.PatchEmbeddingLayer,
            this.model.ClassificationLayer
        }.Concat(this.model.StageTransformerLayers)
             .Concat(this.model.PatchMergingLayers)
             .ToArray());
        }

        // Properties to expose model parameters
        public SwinTransformerModel Model => this.model;
        public int NumStages => this.model.StageDepths.Length;
        public int[] StageDepths => this.model.StageDepths;
    }
}
