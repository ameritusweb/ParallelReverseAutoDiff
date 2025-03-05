using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.Test.EmbeddingTest
{
    public class AttentionEmbeddingProcessor
    {
        private PradOp embeddings;
        private PradOp queryWeight;
        private PradOp keyWeight;
        private PradOp valueWeight;
        private PradOp outputWeight;
        private Random rand;
        private int batchSize = 4;
        private int attentionDim = 64;

        public AttentionEmbeddingProcessor(int numEmbeddings = 20, int embeddingDim = 100)
        {
            embeddings = new PradOp(Tensor.XavierUniform(new int[] { numEmbeddings, embeddingDim }));
            queryWeight = new PradOp(Tensor.XavierUniform(new int[] { embeddingDim, attentionDim }));
            keyWeight = new PradOp(Tensor.XavierUniform(new int[] { embeddingDim, attentionDim }));
            valueWeight = new PradOp(Tensor.XavierUniform(new int[] { embeddingDim, embeddingDim }));
            outputWeight = new PradOp(Tensor.XavierUniform(new int[] { embeddingDim, embeddingDim }));
            rand = new Random(1);
        }

        private PradResult ProcessSingleEmbedding(PradOp allEmbeddings, int targetIdx)
        {
            var allEmbedBranch = allEmbeddings.Branch();
            // Get the target embedding
            var targetEmb = allEmbeddings.Indexer($"{targetIdx}:{targetIdx + 1}", null);

            PradResult otherEmbeddings = null;
            string indexRange;
            if (targetIdx == 0)
            {
                // If target is first, take 1 to end
                indexRange = $"1:{allEmbedBranch.CurrentShape[0]}";
                otherEmbeddings = allEmbedBranch.Indexer(indexRange, null);
            }
            else if (targetIdx == allEmbedBranch.CurrentShape[0] - 1)
            {
                // If target is last, take start to last-1
                indexRange = $"0:{targetIdx}";
                otherEmbeddings = allEmbedBranch.Indexer(indexRange, null);
            }
            else
            {
                // If target is in middle, concatenate both ranges
                var firstPart = $"0:{targetIdx}";
                var secondPart = $"{targetIdx + 1}:{allEmbeddings.Result.Shape[0]}";
                var allBranch = allEmbeddings.Branch();
                var otherEmbeddings1 = allEmbedBranch.Indexer(firstPart, null);
                var otherEmbeddings2 = allBranch.Indexer(secondPart, null);

                // Stack the two parts
                otherEmbeddings = otherEmbeddings1.PradOp.Concat(new[] { otherEmbeddings2.Result });
            }

            var otherEmbedBranch = otherEmbeddings.Branch();

            // Transform embeddings to Q,K,V spaces
            var query = targetEmb.PradOp.MatMul(queryWeight.CurrentTensor);  // [1, attn_dim]
            var keys = otherEmbeddings.PradOp.MatMul(keyWeight.CurrentTensor);  // [batch-1, attn_dim]
            var values = otherEmbedBranch.MatMul(valueWeight.CurrentTensor);  // [batch-1, emb_dim]

            var transpose = keys.PradOp.Transpose(1, 0);

            // Calculate attention scores
            var scores = query.PradOp.MatMul(transpose.Result);  // [1, batch-1]

            // Scale scores and apply softmax
            var scaleFactor = 1.0 / Math.Sqrt(attentionDim);
            var scaledScores = scores.PradOp.Mul(new Tensor(scores.Result.Shape, scaleFactor));

            // Softmax
            var expScores = scaledScores.PradOp.Exp();
            var eScoresBranch = expScores.Branch();
            var sumExp = expScores.PradOp.Sum(new int[] { 1 });
            var attentionWeights = eScoresBranch.Div(sumExp.PradOp.BroadcastTo(expScores.Result.Shape).Result);

            // Apply attention weights to values
            var weightedValues = attentionWeights.PradOp.MatMul(values.Result);  // [1, emb_dim]

            // Final transformation
            return weightedValues.PradOp.MatMul(outputWeight.CurrentTensor);
        }

        private PradResult ProcessAllEmbeddings(PradResult selectedEmbeddings)
        {
            var processedResults = new List<PradResult>();

            // Process each embedding individually
            int count = selectedEmbeddings.Result.Shape[0];
            var branches = selectedEmbeddings.BranchStack(count - 1);
            for (int i = 0; i < count; i++)
            {
                var processed = ProcessSingleEmbedding(i == 0 ? selectedEmbeddings.PradOp : branches.Pop(), i);
                processedResults.Add(processed);
            }

            // Stack all processed embeddings
            return processedResults[0].PradOp.Concat(
                processedResults.Skip(1).Select(r => r.Result).ToArray());
        }

        private Tensor SampleIndices()
        {
            var indices = new double[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                indices[i] = rand.Next(20);
            }
            return new Tensor(new int[] { batchSize }, indices);
        }

        public void Train(double learningRate = 0.01, int epochs = 100)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Sample random indices
                var indices = SampleIndices();
                var pInd = new PradOp(indices);

                // Get selected embeddings
                var selectedEmbeddings = pInd.Embedding(embeddings.CurrentTensor);

                // Process embeddings through attention mechanism
                var processedEmbeddings = ProcessAllEmbeddings(selectedEmbeddings);

                // Initialize loss
                PradResult loss = null;

                var processedEmbBranch = processedEmbeddings.BranchStack(batchSize * batchSize);

                int aaa = 0;
                // Compute distances between processed embeddings
                for (int i = 0; i < batchSize; i++)
                {
                    for (int j = i + 1; j < batchSize; j++)
                    {
                        bool isEmb1Odd = ((int)indices.Data[i]) % 2 == 1;
                        bool isEmb2Odd = ((int)indices.Data[j]) % 2 == 1;

                        if (isEmb1Odd && isEmb2Odd)
                        {
                            aaa++;
                            var op = aaa == 1 ? processedEmbeddings.PradOp : processedEmbBranch.Pop();
                            var emb1 = op.Indexer($"{i}:{i + 1}", null);
                            var emb2 = processedEmbBranch.Pop().Indexer($"{j}:{j + 1}", null);

                            // Attraction loss for odd pairs
                            if (loss == null)
                            {
                                loss = emb1.PradOp.Sub(emb2.Result)
                                                     .PradOp.Square()
                                                     .PradOp.Sum(new int[] { 0, 1 });
                            }
                            else
                            {
                                loss.PradOp.Add(emb1.PradOp.Sub(emb2.Result)
                                                         .PradOp.Square()
                                                         .PradOp.Sum(new int[] { 0, 1 }).Result);
                            }
                        }
                        else if (!isEmb1Odd && !isEmb2Odd)
                        {
                            aaa++;
                            var op = aaa == 1 ? processedEmbeddings.PradOp : processedEmbBranch.Pop();

                            var emb1 = op.Indexer($"{i}:{i + 1}", null);
                            var emb2 = processedEmbBranch.Pop().Indexer($"{j}:{j + 1}", null);

                            // Repulsion loss for even pairs
                            if (loss == null)
                            {
                                var dist = emb1.PradOp.Sub(emb2.Result)
                                                  .PradOp.Square()
                                                  .PradOp.Sum(new int[] { 0, 1 });
                                loss = dist.PradOp.Mul(new Tensor(dist.Result.Shape, -1.0))
                                                  .PradOp.Exp();
                            }
                            else
                            {
                                loss.PradOp.Add(emb1.PradOp.Sub(emb2.Result)
                                                  .PradOp.Square()
                                                  .PradOp.Sum(new int[] { 0, 1 })
                                                  .PradOp.Mul(new Tensor(new int[] { 1 }, -1.0))
                                                  .PradOp.Exp().Result);
                            }
                        }
                    }
                }

                processedEmbBranch.Cleanup();

                double totalLoss = loss.PradOp.CurrentTensor.Data[0];

                // Backpropagate and update
                loss.PradOp.SetUpstreamGradient(new Tensor(loss.PradOp.CurrentShape, 1.0));
                loss.PradOp.Back();

                void UpdateParameter(ref PradOp param)
                {
                    var grad = param.SeedGradient.DeepClone();
                    param.ResetGradient();
                    param = new PradOp(param.CurrentTensor.ElementwiseSub(
                        grad.ElementwiseMultiply(new Tensor(grad.Shape, learningRate))));
                }

                UpdateParameter(ref embeddings);
                UpdateParameter(ref queryWeight);
                UpdateParameter(ref keyWeight);
                UpdateParameter(ref valueWeight);
                UpdateParameter(ref outputWeight);

                if (epoch % 1 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss}");
                }
            }
        }
    }
}
