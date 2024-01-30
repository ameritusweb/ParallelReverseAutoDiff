namespace ParallelReverseAutoDiff.GravNetExample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            VectorFieldNetTrainer trainer = new VectorFieldNetTrainer();
            trainer.Train().Wait();
        }
    }
}