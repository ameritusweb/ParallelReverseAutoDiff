using Xunit;

namespace ParallelReverseAutoDiff.Test.StructuredData
{
    public class StructuredDataGeneratorTest
    {
        [Fact]
        public void TestGenerator()
        {
            StructuredDataGenerator gen = new StructuredDataGenerator(64, 4, 10, 20);
            var results = gen.Forward(0);
            gen.Update();
        }
    }
}
