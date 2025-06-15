# models.py

from sentiment_data import *
from utils import *
import numpy as np
import random
from collections import Counter
from typing import List
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) - {"not", "no", "never", "very"}  # keep sentiment words

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer=indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self,sentence:List[str], add_to_indexer:bool=False)-> Counter:
        features=Counter()
        for word in sentence:
            feature_str=f"UNI={word.lower()}"
            feature_idx=self.indexer.add_and_get_index(feature_str,add=add_to_indexer)
            if feature_idx !=-1:
                features[feature_idx]+=1
        return features


class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        lowered = [word.lower() for word in sentence]  # normalize case

        for i in range(len(lowered) - 1):
            bigram = f"BIGRAM={lowered[i]}|{lowered[i + 1]}"
            idx = self.indexer.add_and_get_index(bigram, add=add_to_indexer)
            if idx != -1:
                features[idx] += 1  # count each bigram once per occurrence

        return features


class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        lowered = [word.lower() for word in sentence]

        # Filter out stopwords except useful ones
        filtered = [w for w in lowered if w not in stop_words]

        # Unigram features
        for word in filtered:
            feat = f"UNI={word}"
            idx = self.indexer.add_and_get_index(feat, add=add_to_indexer)
            if idx != -1:
                features[idx] += 1

        # Bigram features (on filtered sentence)
        for i in range(len(filtered) - 1):
            feat = f"BIGRAM={filtered[i]}|{filtered[i + 1]}"
            idx = self.indexer.add_and_get_index(feat, add=add_to_indexer)
            if idx != -1:
                features[idx] += 1

        return features


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
   
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[idx] * val for idx, val in feats.items())
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[idx] * val for idx, val in feats.items())
        prob = 1.0 / (1.0 + np.exp(-score))  # sigmoid
        return 1 if prob >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor,num_epochs = 30,schedule: str = "constant",) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    #raise Exception("Must be implemented")
   
    indexer = feat_extractor.get_indexer()
    
    random.seed(42)

    # Build vocab
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.zeros(len(indexer))

    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words)
            score = sum(weights[idx] * val for idx, val in feats.items())
            pred = 1 if score >= 0 else 0
            if pred != ex.label:
                for idx, val in feats.items():
                    weights[idx] += val if ex.label == 1 else -val

    return PerceptronClassifier(weights, feat_extractor)

def train_logistic_regression(
    train_exs: List[SentimentExample],
    feat_extractor: FeatureExtractor,
    dev_exs: List[SentimentExample],
    num_epochs: int = 30,
    learning_rate: float = 0.05,
    plot: bool = True
) -> LogisticRegressionClassifier:
    indexer = feat_extractor.get_indexer()
    random.seed(42)

    # Build vocabulary
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.zeros(len(indexer))
    log_likelihoods = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        total_log_likelihood = 0.0
        random.shuffle(train_exs)

        for ex in train_exs:
            feats = feat_extractor.extract_features(ex.words)
            score = sum(weights[idx] * val for idx, val in feats.items())
            prob = 1.0 / (1.0 + np.exp(-score))  # sigmoid

            # Compute log-likelihood
            total_log_likelihood += ex.label * np.log(prob + 1e-12) + (1 - ex.label) * np.log(1 - prob + 1e-12)

            # Gradient update
            error = ex.label - prob
            for idx, val in feats.items():
                weights[idx] += learning_rate * error * val

        avg_ll = total_log_likelihood / len(train_exs)
        log_likelihoods.append(avg_ll)

        classifier = LogisticRegressionClassifier(weights, feat_extractor)
        dev_acc = evaluate_dev_accuracy(classifier, dev_exs)
        dev_accuracies.append(dev_acc)

        print(f"Epoch {epoch + 1} | Avg Log-Likelihood: {avg_ll:.4f} | Dev Accuracy: {dev_acc:.4f}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(log_likelihoods, label="Train Log-Likelihood")
        plt.xlabel("Epoch")
        plt.ylabel("Average Log-Likelihood")
        plt.title("Training Objective")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(dev_accuracies, label="Dev Accuracy", color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Dev Accuracy vs Epoch")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return LogisticRegressionClassifier(weights, feat_extractor)




def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
       # model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs,feat_extractor,dev_exs,num_epochs=30,learning_rate=0.05,plot=True)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model

def evaluate_dev_accuracy(classifier: SentimentClassifier, dev_exs: List[SentimentExample]) -> float:
    correct = 0
    for ex in dev_exs:
        if classifier.predict(ex.words) == ex.label:
            correct += 1
    return correct / len(dev_exs)
