using ParallelReverseAutoDiff.Interprocess;
using ParallelReverseAutoDiff.RMAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class HDF5Test
    {

        [Fact]
        public void GivenADictionaryOfMatrices_SerializesProperly()
        {
            Dictionary<string, (object, object)> elements = new Dictionary<string, (object, object)>();
            for (int j = 0; j < 20; j++)
            {
                DeepMatrix dm1 = new DeepMatrix(2, 1, 1);
                Matrix m1 = new Matrix(500, 500);
                m1.Initialize(InitializationType.Xavier);
                Matrix m2 = new Matrix(500, 500);
                m2.Initialize(InitializationType.Xavier);
                dm1[0] = m1;
                dm1[1] = m2;
                DeepMatrix dm2 = (DeepMatrix)dm1.Clone();
                elements.Add("key1_"+j, (dm1, dm2));

                DeepMatrix dm3 = (DeepMatrix)dm1.Clone();
                DeepMatrix dm4 = (DeepMatrix)dm2.Clone();
                elements.Add("key2_"+j, (dm3, dm4));
            }

            HDF5Helper.Serialize(new FileInfo("mytestfile"), elements, new Func<(object, object), object>[] { x => x.Item1, x => x.Item2 });
        }

        [Fact]
        public void GivenADictionaryOfMatrices_SerializesAndDeserializesProperly()
        {
            Dictionary<string, (object, object)> elements = new Dictionary<string, (object, object)>();
            DeepMatrix dm1 = new DeepMatrix(2, 1, 1);
            Matrix m1 = new Matrix(2, 4);
            m1.Initialize(InitializationType.Xavier);
            Matrix m2 = new Matrix(2, 4);
            m2.Initialize(InitializationType.Xavier);
            dm1[0] = m1;
            dm1[1] = m2;
            DeepMatrix dm2 = (DeepMatrix)dm1.Clone();
            elements.Add("key1", (dm1, dm2));

            DeepMatrix dm3 = (DeepMatrix)dm1.Clone();
            DeepMatrix dm4 = (DeepMatrix)dm2.Clone();
            elements.Add("key2", (dm3, dm4));

            HDF5Helper.Serialize(new FileInfo("mytestfile"), elements, new Func<(object, object), object>[] { x => x.Item1, x => x.Item2 });
            var matrices = HDF5Helper.Deserialize(new FileInfo("mytestfile.bin"));
            Assert.Equal(m1.ToArray(), matrices.Take(2).ToArray());

            (elements["key1"].Item1 as DeepMatrix)![0].Initialize(InitializationType.Xavier);
            (elements["key1"].Item1 as DeepMatrix)![1].Initialize(InitializationType.Xavier);

            HDF5Helper.Deserialize(new FileInfo("mytestfile"), elements, new Func<(object, object), object>[] { x => x.Item1, x => x.Item2 });
            Assert.Equal(m1.ToArray(), (elements["key1"].Item1 as DeepMatrix)![0].ToArray());
        }
    }
}