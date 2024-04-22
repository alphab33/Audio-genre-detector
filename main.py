from audio_genre_classifier import AudioGenreClassifier

def main():
    classifier = AudioGenreClassifier()
    
    # Load data
    tracks, echonest_metrics = classifier.load_data()
    
    # Preprocess data
    features, labels = classifier.preprocess_data(tracks, echonest_metrics)
    
    # Train and evaluate models
    classifier.train_and_evaluate_models(features, labels)

if __name__ == "__main__":
    main()
