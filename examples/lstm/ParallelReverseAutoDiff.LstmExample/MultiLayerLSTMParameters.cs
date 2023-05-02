namespace ParallelReverseAutoDiff.LstmExample
{
    [Serializable]
    public class MultiLayerLSTMParameters
    {
        public double[][][] Wi, Wf, Wc, Wo;
        public double[][][] Ui, Uf, Uc, Uo;
        public double[][][] bi, bf, bc, bo;
        public double[][] be, We, V, b;
        public double[][][] Wq, Wk, Wv;
    }
}
