using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using binaryClassification;
using Microsoft.ML;
using Microsoft.ML.Data;
using SentimentalAnal.Models;

namespace SentimentalAnal
{
    public static class BeSentimental
    {
        private static readonly string _dataPath =
        Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
        public static void Sentimental(List<CryptoHit> ch)
        {
            var mlContext = new MLContext();
            var splitDataView = LoadData(mlContext);
            var model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Evaluate(mlContext, model, splitDataView.TestSet);
            UseModelWithBatchItems(mlContext, model, ch);
        }


        /// <summary>
        /// Loads the data.
        /// Splits the loaded dataset into train and test datasets.
        /// Returns the split train and test datasets.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private static DataOperationsCatalog.TrainTestData LoadData(MLContext mlContext)
        {
            var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            var splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        /// <summary>
        /// Extracts and transforms the data.
        /// Trains the model.
        /// Predicts sentiment based on test data.
        /// Returns the model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="trainSet"></param>
        /// <returns></returns>
        private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            return model;
        }

        /// <summary>
        /// Loads the test dataset.
        /// Creates the BinaryClassification evaluator.
        /// Evaluates the model and creates metrics.
        /// Displays the metrics.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        /// <param name="splitTestSet"></param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }
        
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model, List<CryptoHit> ch)
        {
            var cryptolist = ch.GroupBy(x => x.Name).Select(x => x).ToList();
            var cs = new List<CryptoScore>();
            foreach (var article in cryptolist)
            {
                var sentiments = new List<SentimentData>();
                foreach (CryptoHit cryptoHit in article)
                {
                    sentiments.Add(new SentimentData
                    {
                        SentimentText = cryptoHit.Title
                    });
                }

                IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
                IDataView predictions = model.Transform(batchComments);
                // Use model to predict whether comment data is Positive (1) or Negative (0).
                IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
                Console.WriteLine();
                Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
                var ts = new List<TheSentiment>();
                foreach (SentimentPrediction prediction in predictedResults)
                {
                    ts.Add(new TheSentiment
                    {
                        Prediction = Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative", 
                        Probability = prediction.Probability,
                        Sentiment = prediction.SentimentText
                    });
                    Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
                }
                Console.WriteLine("=============== End of predictions ===============");
                cs.Add(new CryptoScore
                {
                    Crypto = article.Key,
                    Sentiment = ts
                });
            }

            var a = from c in cs
                select new
                {
                    c.Crypto,
                    V = c.Sentiment.Max(x => x.Prediction)
                };
            
            Console.WriteLine();
        }
    }
}
