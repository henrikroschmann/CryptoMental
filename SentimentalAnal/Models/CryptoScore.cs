using System.Collections.Generic;
using binaryClassification;

namespace SentimentalAnal.Models
{
    internal class CryptoScore
    {
        public Crypto Crypto { get; set; }
        public List<TheSentiment> Sentiment { get; set; }
    }
}
