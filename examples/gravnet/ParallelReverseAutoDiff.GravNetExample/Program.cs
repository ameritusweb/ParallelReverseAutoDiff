namespace ParallelReverseAutoDiff.GravNetExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            SyllableExtractor extrator = new SyllableExtractor();
            extrator.Extract();
            var syll = Syllabifier.Syllabify("through");

            // VectorFieldNetTrainer trainer = new VectorFieldNetTrainer();
            // trainer.Train().Wait();
        }
    }
}