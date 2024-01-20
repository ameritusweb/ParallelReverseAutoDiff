namespace ParallelReverseAutoDiff.GravNetExample
{
    public class RepeatToken
    {
        private int _remainingRepetitions;
        public bool ShouldRepeat => _remainingRepetitions > 0;
        public int UsageCount { get; private set; } = 0;
        public int TotalRepeatCount { get; private set; } = 0;
        public bool ShouldContinue { get; private set; }

        public void Repeat(int times = 1)
        {
            if (times < 1)
            {
                throw new ArgumentException("Repeat times must be at least 1.", nameof(times));
            }
            _remainingRepetitions = times;
            UsageCount++;
            TotalRepeatCount += times;
        }

        public void Decrement()
        {
            if (_remainingRepetitions > 0)
            {
                _remainingRepetitions--;
            }
        }

        public void Continue()
        {
            ShouldContinue = true;
        }

        public void Reset()
        {
            ShouldContinue = false;
        }
    }

}
