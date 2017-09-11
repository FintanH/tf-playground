{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE RecordWildCards #-}

module Iris
    ( runIris
    , readIrisFromFile
    ) where

import           Control.Monad              (forM_, when)
import           Control.Monad.IO.Class     (liftIO)
import           Control.Monad.Trans.Either (EitherT, hoistEither)
import           Data.ByteString.Lazy.Char8 (pack)
import           Data.Csv                   (FromRecord, HasHeader (NoHeader),
                                             decode)
import           Data.Int                   (Int64)
import           Data.Vector                (Vector, fromList, (!))
import qualified Data.Vector                as V
import           GHC.Generics               (Generic)
import           System.Random.Shuffle      (shuffleM)
import           TensorFlow.Core            (Build, Session, Tensor, TensorData,
                                             build, encodeTensorData, feed,
                                             render, runSession, runWithFeeds,
                                             unScalar)
import           TensorFlow.Minimize        (adam, minimizeWith)
import           TensorFlow.Ops             (add, argMax, cast, equal, matMul,
                                             oneHot, placeholder, reduceMean,
                                             relu, scalar,
                                             softmaxCrossEntropyWithLogits,
                                             truncatedNormal, vector)
import           TensorFlow.Variable        (Variable, initializedVariable,
                                             readValue)

type FeatureLength = Int64
type LabelsLength = Int64

-- | Each line of the CSV contains 4 features followed by a label
-- | We capture the four features in the IrisRecord type
data IrisRecord = IrisRecord {
    feature1 :: Float
  , feature2 :: Float
  , feature3 :: Float
  , feature4 :: Float
  , label    :: Int64
} deriving (Show, Generic)

instance FromRecord IrisRecord

-- | Helper for reading in an Iris CSV
readIrisFromFile :: FilePath -> EitherT String IO (Vector IrisRecord)
readIrisFromFile fp = do
  contents <- liftIO $ readFile fp
  hoistEither $ decode NoHeader (pack contents)

-- | Choose a random sample of records
chooseRandomRecords :: Int -> Vector IrisRecord -> IO (Vector IrisRecord)
chooseRandomRecords sampleSize records = do
  let numRecords = V.length records
  chosenIndices <- take sampleSize <$> shuffleM [0..numRecords - 1]
  pure . fromList . map (records !) $ chosenIndices

convertRecordsToTensorData :: FeatureLength
                           -> Vector IrisRecord
                           -> (TensorData Float, TensorData Int64)
convertRecordsToTensorData featureLength records = (input, output)
  where
    numRecords = V.length records

    -- | Input is a matrix NxM where N is the number of records and M is
    -- | is the number of features
    input = encodeTensorData [fromIntegral numRecords, featureLength] (fromList $ concatMap recordToFeat records)

    -- | Our output is a vector of size N, each entry a label
    output = encodeTensorData [fromIntegral numRecords] (label <$> records)

    -- | Helper to extract the features of our record
    recordToFeat :: IrisRecord -> [Float]
    recordToFeat IrisRecord{..} = [feature1, feature2, feature3, feature4]

-- | Our Model type for training and getting the error rate
-- | Two steps
-- | Step one: take our training data and train a set of weights
-- | Step Two: take our test set and determine our error rate
data Model = Model {
    train     :: TensorData Float -- Training input
              -> TensorData Int64 -- Training output
              -> Session ()
  , errorRate :: TensorData Float -- Test input
              -> TensorData Int64 -- Test output
              -> Session Float
}

buildNNLayer :: Int64 -> Int64 -> Tensor v Float
             -> Build (Variable Float, Variable Float, Tensor Build Float)
buildNNLayer inputSize outputSize input = do
  weights <- truncatedNormal (vector [inputSize, outputSize]) >>= initializedVariable
  bias <- truncatedNormal (vector [outputSize]) >>= initializedVariable
  let results = (input `matMul` readValue weights) `add` readValue bias
  pure (weights, bias, results)

createModel :: FeatureLength -> LabelsLength -> Build Model
createModel featureLength labelsLength = do
  let batchSize = -1
      numHiddenUnits = 10
  inputs <- placeholder [batchSize, featureLength]
  outputs <- placeholder [batchSize]
  (hiddenWeights, hiddenBiases, hiddenResults) <- buildNNLayer featureLength numHiddenUnits inputs

  let rectifiedHiddenResults = relu hiddenResults
  (finalWeights, finalBiases, finalResults) <- buildNNLayer numHiddenUnits labelsLength rectifiedHiddenResults
  actualOutput <- render $ cast $ argMax finalResults (scalar (1 :: Int64))

  let correctedPredictions = equal actualOutput outputs
  errorRate_ <- render $ 1 - reduceMean (cast correctedPredictions)

  let outputVectors = oneHot outputs (fromIntegral labelsLength) 1 0
      loss = reduceMean . fst $ softmaxCrossEntropyWithLogits finalResults outputVectors
      params = [hiddenWeights, hiddenBiases, finalWeights, finalBiases]
  train_ <- minimizeWith adam loss params

  pure Model
    { train = \inputFeed outputFeed ->
        runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          train_
    , errorRate = \inputFeed outputFeed -> unScalar <$>
        runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          errorRate_
    }

runIris :: Int -> FeatureLength -> LabelsLength -> Vector IrisRecord -> Vector IrisRecord -> IO ()
runIris sampleSize featureLength labelsLength trainingRecords testRecords =
  runSession $ do
    -- Preparation
    model <- build (createModel featureLength labelsLength)

    -- Training
    forM_ ([0..1000] :: [Int]) $ \i -> do
      trainingSample <- liftIO $ chooseRandomRecords sampleSize trainingRecords
      let (trainingInputs, trainingOutputs) = convertRecordsToTensorData featureLength trainingSample
      train model trainingInputs trainingOutputs
      when (i `mod` 100 == 0) $ do
        err <- errorRate model trainingInputs trainingOutputs
        liftIO $ putStrLn $ "Current training error " ++ show (err * 100)

    liftIO $ putStrLn ""

    -- Testing
    let (testingInputs, testingOutputs) = convertRecordsToTensorData featureLength testRecords
    testingError <- errorRate model testingInputs testingOutputs
    liftIO $ putStrLn $ "test error " ++ show (testingError * 100)
