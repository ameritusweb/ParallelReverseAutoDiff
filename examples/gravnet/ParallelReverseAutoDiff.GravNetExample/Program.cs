namespace ParallelReverseAutoDiff.GravNetExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            GravNetTrainer trainer = new GravNetTrainer();
            trainer.Train().Wait();
        }
    }
}