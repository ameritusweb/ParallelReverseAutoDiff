// ------------------------------------------------------------------------------
// <copyright file="EmbeddingNeuralNetwork.cs" author="ameritusweb" date="6/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.Test.GraphAttentionPaths.Embedding
{
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.GnnExample.Common;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// An embedding neural network.
    /// </summary>
    public class EmbeddingNeuralNetwork : NeuralNetwork
    {
        private const string NAMESPACE = "ParallelReverseAutoDiff.GnnExample.GraphAttentionPaths.Embedding.Architecture";
        private const string ARCHITECTURE = "Embedding";

        private readonly IModelLayer embeddingLayer;

        private EmbeddingComputationGraph computationGraph;

        /// <summary>
        /// Initializes a new instance of the <see cref="EmbeddingNeuralNetwork"/> class.
        /// </summary>
        /// <param name="numIndices">The number of indices.</param>
        /// <param name="alphabetSize">The alphabet size.</param>
        /// <param name="embeddingSize">The embedding size.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="clipValue">The clip value.</param>
        public EmbeddingNeuralNetwork(int numIndices, int alphabetSize, int embeddingSize, double learningRate, double clipValue)
        {
            this.Parameters.LearningRate = learningRate;
            this.Parameters.ClipValue = clipValue;
            this.AlphabetSize = alphabetSize;
            this.NumIndices = numIndices;
            this.EmbeddingSize = embeddingSize;

            this.embeddingLayer = new ModelLayerBuilder(this)
                .AddModelElementGroup("Embeddings", new[] { alphabetSize, embeddingSize }, InitializationType.Xavier)
                .Build();

            this.InitializeState();
        }

        /// <summary>
        /// Gets the input matrix.
        /// </summary>
        public DeepMatrix Input { get; private set; }

        /// <summary>
        /// Gets the hand picked features.
        /// </summary>
        public DeepMatrix HandPickedFeatures { get; private set; }

        /// <summary>
        /// Gets the output matrix.
        /// </summary>
        public DeepMatrix Output { get; private set; }

        /// <summary>
        /// Gets the target matrix.
        /// </summary>
        public Matrix Target { get; private set; }

        /// <summary>
        /// Gets the model layers.
        /// </summary>
        public IEnumerable<IModelLayer> ModelLayers
        {
            get
            {
                return new IModelLayer[] { this.embeddingLayer }.ToList();
            }
        }

        /// <summary>
        /// Gets the alphabet size of the neural network.
        /// </summary>
        internal int AlphabetSize { get; private set; }

        /// <summary>
        /// Gets the number of indices of the neural network.
        /// </summary>
        internal int NumIndices { get; private set; }

        /// <summary>
        /// Gets the embedding size of the neural network.
        /// </summary>
        internal int EmbeddingSize { get; private set; }

        /// <summary>
        /// Initializes the computation graph of the convolutional neural network.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task Initialize()
        {
            await this.InitializeComputationGraph();
        }

        /// <summary>
        /// Store the operation intermediates.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void StoreOperationIntermediates(Guid id)
        {
            this.computationGraph.StoreOperationIntermediates(id);
        }

        /// <summary>
        /// Restore the operation intermediates.
        /// </summary>
        /// <param name="id">The identifier.</param>
        public void RestoreOperationIntermediates(Guid id)
        {
            this.computationGraph.RestoreOperationIntermediates(id);
        }

        /// <summary>
        /// The forward pass of the edge attention neural network.
        /// </summary>
        /// <param name="input">The input.</param>
        public void AutomaticForwardPropagate(DeepMatrix input)
        {
            // Initialize hidden state, gradients, biases, and intermediates
            this.ClearState();

            CommonMatrixUtils.SetInPlaceReplace(this.Input, input);
            var op = this.computationGraph.StartOperation;
            if (op == null)
            {
                throw new Exception("Start operation should not be null.");
            }

            IOperationBase? currOp = null;
            do
            {
                var parameters = this.LookupParameters(op);

                var forward = op.OperationType.GetMethod("Forward", parameters.Select(x => x.GetType()).ToArray());
                if (forward == null)
                {
                    throw new Exception($"Forward method not found for operation {op.OperationType.Name}");
                }

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

        /// <summary>
        /// The backward pass of the edge attention neural network.
        /// </summary>
        /// <param name="gradient">The gradient of the loss.</param>
        /// <returns>The gradient.</returns>
        public async Task<DeepMatrix> AutomaticBackwardPropagate(DeepMatrix gradient)
        {
            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["vector_concatenate_trans_0_0"];
            if (!CommonMatrixUtils.IsAllZeroes(gradient))
            {
                backwardStartOperation.BackwardInput = gradient;
                OperationNeuralNetworkVisitor opVisitor = new OperationNeuralNetworkVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
                opVisitor.RunSequentially = false;
                await opVisitor.TraverseAsync();
                if (opVisitor.AggregateException != null)
                {
                    if (opVisitor.AggregateException.InnerExceptions.Count > 1)
                    {
                        throw opVisitor.AggregateException;
                    }
                    else
                    {
                        Console.WriteLine(opVisitor.AggregateException.InnerExceptions[0].Message);
                    }
                }

                opVisitor.Reset();
            }

            IOperationBase? backwardEndOperation = this.computationGraph["batch_embeddings_0_0"];
            return backwardEndOperation.CalculatedGradient[0] as DeepMatrix ?? throw new InvalidOperationException("Calculated gradient should not be null.");
        }

        /// <summary>
        /// Initialize the state of the edge attention neural network.
        /// </summary>
        public void InitializeState()
        {
            if (this.Output == null)
            {
                this.Output = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, (this.NumIndices * this.EmbeddingSize) + 3, 1));
            }
            else
            {
                this.Output.Replace(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, (this.NumIndices * this.EmbeddingSize) + 3, 1));
            }

            if (this.HandPickedFeatures == null)
            {
                this.HandPickedFeatures = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, 1, 3));
            }
            else
            {
                this.HandPickedFeatures.Replace(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, 1, 3));
            }

            if (this.Input == null)
            {
                this.Input = new DeepMatrix(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumIndices, 1));
            }
            else
            {
                this.Input.Replace(CommonMatrixUtils.InitializeZeroMatrix(this.Parameters.BatchSize, this.NumIndices, 1));
            }
        }

        /// <summary>
        /// Clear the state of the edge attention neural network.
        /// </summary>
        private void ClearState()
        {
            GradientClearer clearer = new GradientClearer();
            clearer.Clear(new[] { this.embeddingLayer });
        }

        /// <summary>
        /// Initialize the computation graph of the edge attention neural network.
        /// </summary>
        /// <returns>A task.</returns>
        private async Task InitializeComputationGraph()
        {
            var weightMatrix = this.embeddingLayer.WeightMatrix("Embeddings");
            var gradientMatrix = this.embeddingLayer.GradientMatrix("Embeddings");

            string json = EmbeddedResource.ReadAllJson(NAMESPACE, ARCHITECTURE);
            var jsonArchitecture = JsonConvert.DeserializeObject<JsonArchitecture>(json) ?? throw new InvalidOperationException("There was a problem deserialzing the JSON architecture.");
            this.computationGraph = new EmbeddingComputationGraph(this);
            this.computationGraph
                .AddIntermediate("Input", _ => this.Input)
                .AddIntermediate("Output", _ => this.Output)
                .AddIntermediate("HandPickedFeatures", _ => this.HandPickedFeatures)
                .AddWeight("Embeddings", _ => weightMatrix).AddGradient("DEmbeddings", _ => gradientMatrix)
                .ConstructFromArchitecture(jsonArchitecture);

            IOperationBase? backwardStartOperation = null;
            backwardStartOperation = this.computationGraph["vector_concatenate_trans_0_0"];
            OperationGraphVisitor opVisitor = new OperationGraphVisitor(Guid.NewGuid().ToString(), backwardStartOperation, 0);
            await opVisitor.TraverseAsync();
            await opVisitor.ResetVisitedCountsAsync(backwardStartOperation);
        }
    }
}
