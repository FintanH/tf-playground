{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import           Control.Monad              (replicateM_)
import           Control.Monad.IO.Class     (liftIO)
import           Control.Monad.Trans.Either (runEitherT)
import           Data.Vector                (Vector)
import qualified Data.Vector                as Vector
import           Iris
import           TensorFlow.Core            (Scalar (..), Tensor, Value,
                                             encodeTensorData, feed)
import           TensorFlow.GenOps.Core     (square)
import           TensorFlow.Minimize        (gradientDescent, minimizeWith)
import           TensorFlow.Ops             (add, mul, placeholder, reduceSum,
                                             sub)
import           TensorFlow.Session         (run, runSession, runWithFeeds)
import           TensorFlow.Variable        (Variable, initializedVariable,
                                             readValue)

main :: IO ()
main = do
  let irisFeatures = 4
      irisLabels = 3
      sampleSize = 10
  result <- runEitherT $ do
    trainRecords <- readIrisFromFile "data/iris_train.csv"
    testRecords <- readIrisFromFile "data/iris_test.csv"
    liftIO $ runIris sampleSize irisFeatures irisLabels trainRecords testRecords
  case result of
    Left err -> putStrLn err
    Right r -> pure ()
  -- results <- basicExample
  --     [1.0, 2.0, 3.0, 4.0]
  --     [4.0, 9.0, 14.0, 19.0]
  -- print results

-- Taking two vectors (should be of equal length) and learn the outputs
-- for a linear equation
basicExample :: Vector Float -> Vector Float -> IO (Float, Float)
basicExample xInput yInput = runSession $ do
  -- Get the sizes of the two vector inputs
  let xSize = vectorSize xInput
      ySize = vectorSize yInput

  -- Make a "weight" variable (i.e. the slope of the line) with initial value 3
  (w :: Variable Float) <- initializedVariable 3
  -- Make a "bias" variable (y-intercept of line) with initial value 1
  (b :: Variable Float) <- initializedVariable 1

  -- Create "placeholders" with the size of our input and output
  (x :: Tensor Value Float) <- placeholder [xSize]
  (y :: Tensor Value Float) <- placeholder [ySize]

  -- Make our "model"
  -- Multiplies the weights `w` by the input `x` and then adds the bias `b`.
  -- The `readValue` operations give us our "actual output" value.
  let linearModel = (readValue w `mul` x) `add` readValue b
  -- Get the difference between our learned model `linearModel`
  -- and our expected output `y` and squaring it
      squareDeltas = square $ linearModel `sub` y
  -- Our "loss" fucntion is the reduced sum of of our squared differences
      loss = reduceSum squareDeltas

  -- We "train" the model using gradient descent optimizer passing in the
  -- weights and bias as the parameters that change
  trainStep <- minimizeWith (gradientDescent 0.01) loss [w, b]

  -- "Train" our model, passing input and output values as "feeds" to fill in
  -- the placeholder values
  let trainWithFeeds xF yF = runWithFeeds [ feed x xF
                                          , feed y yF ]
                                          trainStep
  -- Runs the training 1000 times and encodes inputs as "TensorData"
  replicateM_ 1000 $
    trainWithFeeds (encodeTensorData [xSize] xInput)
                   (encodeTensorData [ySize] yInput)

  -- We "run" our variables to see the learned values and return them
  (Scalar wLearned, Scalar bLearned) <- run (readValue w, readValue b)
  pure (wLearned, bLearned)
  where
    vectorSize = fromIntegral . Vector.length
