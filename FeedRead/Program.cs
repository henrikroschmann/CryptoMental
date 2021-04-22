using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.ServiceModel.Syndication;
using System.Xml;
using FeedRead.Models;
using SentimentalAnal;
using SentimentalAnal.Models;

namespace FeedRead
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var CryptoHists = new Dictionary<Crypto, string[]>();
            CryptoHists.Add(Crypto.XRP, new string[] { "ripple", "xrp"});
            CryptoHists.Add(Crypto.BTC, new string[] { "btc", "bitcoin"});
            CryptoHists.Add(Crypto.XLM, new string[] { "stellar lumens", "xlm" });
            CryptoHists.Add(Crypto.ETH, new string[] { "eth", "ethereum"});
            CryptoHists.Add(Crypto.BNB, new string[] { "bnb", "binance coin" });
            CryptoHists.Add(Crypto.LTC, new string[] { "ltc", "litecoin"});

            var feedList = new List<Feed>();
            var lines = System.IO.File.ReadAllLines(
                Path.Combine(Environment.CurrentDirectory, "Data", "Feeds.txt"));
            foreach (var line in lines)
            {
                try
                {
                    var reader = XmlReader.Create(line);
                    var feed = SyndicationFeed.Load(reader);
                    reader.Close();
                    var art = feed.Items.Select(item => new Article { Title = item.Title.Text, DateTime = item.PublishDate.DateTime }).ToList();
                    feedList.Add(new Feed
                    {
                        Source = line,
                        Title = art
                    });
                }
                catch (Exception)
                {
                    Console.WriteLine($"cannot parse feed moving on {line}");
                }
            }

            var articleList = from feed in feedList
                from article in feed.Title
                where article.DateTime.Date >= DateTime.Parse("2021-04-22")
                select article.Title;

            var ch = new List<CryptoHit>();
            foreach (var article in articleList)
            {
                foreach (var cryptoHist in CryptoHists)
                {
                    if (!cryptoHist.Value.Any(x => article.ToLower().Contains(x))) continue;
                    ch.Add(new CryptoHit
                    {
                        Name = cryptoHist.Key,
                        Title = article
                    });
                }
                
            }

            BeSentimental.Sentimental(ch);
        }
    }
}