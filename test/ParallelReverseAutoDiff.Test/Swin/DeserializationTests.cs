using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Common;
using Xunit;

namespace ParallelReverseAutoDiff.Test.Swin
{
    public class DeserializationTests
    {
        [Fact]
        public void TestDeserialization()
        {
            string json = EmbeddedResource.ReadAllJson("ParallelReverseAutoDiff.Test.Swin", "swin-transformer");
            var jsonArchitecture = JsonConvert.DeserializeObject<FourLayersJsonArchitecture>(json);
        }
    }
}
