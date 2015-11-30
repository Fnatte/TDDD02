#!/usr/bin/python3

# Skeleton code for the assignment on text classification
# Marco Kuhlmann <marco.kuhlmann@liu.se>

import nb
import pprint
from math import log

pp = pprint.PrettyPrinter(indent=4)

# List of stop words

STOP_WORDS = set("och det att i en jag hon som han på den med var sig för så till är men ett om hade de av icke mig du henne då sin nu har inte hans honom skulle hennes där min man ej vid kunde något från ut när efter upp vi dem vara vad över än dig kan sina här ha mot alla under någon eller allt mycket sedan ju denna själv detta åt utan varit hur ingen mitt ni bli blev oss din dessa några deras blir mina samma vilken er sådan vår blivit dess inom mellan sådant varför varje vilka ditt vem vilket sitta sådana vart dina vars vårt våra ert era vilkas".split())

# Python class implementing a Naive Bayes classifier.  Initially, the
# methods of this class only call the corresponding methods from the
# superclass (NaiveBayesClassifier).  Your task in the assignment is
# to replace these calls with your own code.

class MyNaiveBayesClassifier(nb.NaiveBayesClassifier):

    def get_tokens(self, speech):
        """Returns the token list for the specified speech."""
        return [token for token in speech['tokenlista'] if token not in STOP_WORDS]

    def get_class(self, speech):
        """Returns the class of the specified speech."""
        return "L" if speech['parti'] in ["MP", "S", "V"] else "R"

    def accuracy(self, speeches):
        """Computes accuracy on the specified test data."""
        correct = 0
        for speech in speeches:
            if self.get_class(speech) == self.predict(speech):
                correct += 1

        return correct / len(speeches)

    def precision(self, c, speeches):
        """Computes precision for class `c` on the specified test data."""
        # Original implementation
        #return super().precision(c, speeches)

        distribution = self.speech_prediction_distribution(c, speeches)

        true_positives = distribution[0]
        true_negatives = distribution[1]
        false_positives = distribution[2]
        false_negatives = distribution[3]

        return true_positives / (true_positives + false_positives)

    def recall(self, c, speeches):
        """Computes recall for class `c` on the specified test data."""
        # Original implementaiton
        # return super().recall(c, speeches)

        distribution = self.speech_prediction_distribution(c, speeches)

        true_positives = distribution[0]
        true_negatives = distribution[1]
        false_positives = distribution[2]
        false_negatives = distribution[3]

        return true_positives / (true_positives + false_negatives)

    def predict(self, speech):
        """Predicts the class of the specified speech."""
        return 'R' if self.class_probability('R', speech) > self.class_probability('L', speech) else 'L'

    def class_probability(self, c, speech):
        words = self.get_tokens(speech)
        probability = self.pc[c]

        for word in words:
            if word in self.pw[c]:
                probability += self.pw[c][word]

        return probability

    def train(self, speeches):
        """Trains using the specified training data."""
        classes = ['R', 'L']
        class_counts = { 'R': 0, 'L': 0}
        word_counts = { 'R': {}, 'L': {} }

        # Count the frequency of each class.
        for speech in speeches:
            class_counts[self.get_class(speech)] += 1

        # Assign P(c) of each class to self.pc
        for c in classes:
            self.pc[c] = log(class_counts[c] / len(speeches))

        # Count the frequency of each word in each class.
        for speech in speeches:
            c = self.get_class(speech)

            for word in self.get_tokens(speech):
                if word in word_counts[c]:
                    word_counts[c][word] += 1
                else:
                    word_counts[c][word] = 1

        # Assign P(w|c) of each word in each class to self.pw
        for c in classes:
            if not c in self.pw:
                self.pw[c] = {}

            num_tokens = sum(word_counts[c].values())
            for word in word_counts[c]:
                self.pw[c][word] = log((word_counts[c][word] + 1) / (num_tokens + len(word_counts)))

    def speech_prediction_distribution(self, c, speeches):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for speech in speeches:
            if c == self.get_class(speech):
                if self.get_class(speech) == self.predict(speech):
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if self.get_class(speech) == self.predict(speech):
                    true_negatives += 1
                else:
                    false_positives += 1

        return (true_positives, true_negatives, false_positives, false_negatives)


# The following code will be run when you call this script from the
# command line.

if __name__ == "__main__":
    import json
    import sys

    def LOG(msg):
        sys.stdout.write(msg)
        sys.stdout.flush()

    # Train a model on training data and save it to a file.
    # Usage: python lab1.py train TRAINING_DATA_FILE MODEL_FILE
    if sys.argv[1] == "train":
        classifier = MyNaiveBayesClassifier()
        with open(sys.argv[2]) as fp:
            LOG("Loading training data from %s ..." % sys.argv[2])
            training_data = json.load(fp)
            LOG(" done\n")
        LOG("Training ...")
        classifier.train(training_data)
        LOG(" done\n")
        LOG("Saving model to %s ..." % sys.argv[3])
        classifier.save(sys.argv[3])
        LOG(" done\n")

    # Load a trained model from a file and evaluate it on test data.
    # Usage: python lab1.py evaluate MODEL_FILE TEST_DATA_FILE
    if sys.argv[1] == "evaluate":
        classifier = MyNaiveBayesClassifier()
        LOG("Loading the model from %s ..." % sys.argv[2])
        classifier.load(sys.argv[2])
        LOG(" done\n")
        with open(sys.argv[3]) as fp:
            LOG("Loading test data from %s ..." % sys.argv[3])
            test_data = json.load(fp)
            LOG(" done\n")
        LOG("accuracy = %.4f\n" % classifier.accuracy(test_data))
        for c in sorted(classifier.pc):
            p = classifier.precision(c, test_data)
            r = classifier.recall(c, test_data)
            LOG("class %s: precision = %.4f, recall = %.4f\n" % (c, p, r))
