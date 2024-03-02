namespace ParallelReverseAutoDiff.GravNetExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            SpatialNetTrainer trainer = new SpatialNetTrainer();
            trainer.Train().Wait();
        }
    }
}