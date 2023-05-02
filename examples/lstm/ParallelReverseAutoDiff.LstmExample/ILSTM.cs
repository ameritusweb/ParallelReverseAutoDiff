namespace ParallelReverseAutoDiff.LstmExample
{
    public interface ILSTM
    {
        double[] GetOutput(double[][][] inputs);

        Task Optimize(double[][][] inputs, List<double[][]> chosenActions, List<double> rewards, int iterationIndex, bool doNotUpdate = false);

        void SaveModel(string path);

        string Name { get; }
    }
}
