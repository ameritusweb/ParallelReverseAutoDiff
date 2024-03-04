namespace ParallelReverseAutoDiff.GravNetExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            GlyphNetTrainer trainer = new GlyphNetTrainer();
            trainer.Train().Wait();
        }
    }
}