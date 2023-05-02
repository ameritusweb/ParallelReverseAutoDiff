namespace ParallelReverseAutoDiff.LstmExample
{
    using System.Reflection;

    public static class EmbeddedResource
    {
        public static string ReadAllJson(string file)
        {
            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = $"ParallelReverseAutoDiff.LstmExample.architecture.{file}.json";

            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            using (StreamReader reader = new StreamReader(stream))
            {
                string result = reader.ReadToEnd();
                return result;
            }
        }
    }
}
