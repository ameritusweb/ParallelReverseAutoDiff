using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace ParallelReverseAutoDiff.Test.EmbeddingTest
{
    public class EmbeddingTests
    {
        [Fact]
        public void TestEmbedding()
        {
            AttentionEmbeddingProcessor processor = new AttentionEmbeddingProcessor();
            processor.Train();
        }
    }
}
