namespace ParallelReverseAutoDiff.GravNetExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            TiledNetTrainer trainer = new TiledNetTrainer();
            trainer.Train().Wait();
        }
    }
}