import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class AudioGenreClassifier:
    def __init__(self):
        pass
    
    #change .csv file to choose from different dataset
    def load_data(self):
        tracks = pd.read_csv('fma-rock-vs-hiphop.csv')
        echonest_metrics = pd.read_json('echonest-metrics.json', precise_float=True)
        return tracks, echonest_metrics
    
    def preprocess_data(self, tracks, echonest_metrics):
        echo_tracks = echonest_metrics.merge(tracks[['track_id', 'genre_top']], on='track_id')
        features = echo_tracks.drop(['genre_top', 'track_id'], axis=1)
        labels = echo_tracks['genre_top']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        pca = PCA(n_components=6, random_state=10)
        pca_projection = pca.fit_transform(scaled_features)
        return pca_projection, labels
    
    def train_and_evaluate_models(self, features, labels):
        # Split data
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=10)
        
        # Train decision tree
        tree = DecisionTreeClassifier(random_state=10)
        tree.fit(train_features, train_labels)
        pred_labels_tree = tree.predict(test_features)
        
        # Train logistic regression
        logreg = LogisticRegression(random_state=10)
        logreg.fit(train_features, train_labels)
        pred_labels_logit = logreg.predict(test_features)
        
        # Print classification reports
        print("Decision Tree:")
        print(classification_report(test_labels, pred_labels_tree))
        print("Logistic Regression:")
        print(classification_report(test_labels, pred_labels_logit))
        
        # Cross-validation
        kf = KFold(10, random_state=10)
        tree_scores = cross_val_score(tree, features, labels, cv=kf)
        logit_scores = cross_val_score(logreg, features, labels, cv=kf)
        print("Decision Tree Cross-Validation Scores:", tree_scores)
        print("Logistic Regression Cross-Validation Scores:", logit_scores)
