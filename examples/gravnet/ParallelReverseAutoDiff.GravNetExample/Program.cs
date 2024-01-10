namespace ParallelReverseAutoDiff.GravNetExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            VectorNetTrainer trainer = new VectorNetTrainer();
            trainer.Train().Wait();
        }
    }
}