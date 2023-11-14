import spacy
import os
import random
from spacy.util import minibatch, compounding
from datasets import load_dataset

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load('en_core_web_sm')
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            'textcat', config={'architecture': 'simple_cnn'}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe('textcat')
    textcat.add_label('positive')
    textcat.add_label('neutral')
    textcat.add_label('negative')
    
    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != 'textcat'
    ]
    
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )
        for i in range(iterations):
            loss = {}
            # random.shuffle(training_data)
            batches = minibatch(
                training_data, size=batch_sizes
            )
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text, labels, drop=0.2, sgd=optimizer,
                    losses=loss
                )
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )
                
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")        
                
def test_model(input_data: str = "This movie sucked"):
    #load saved trained model
    loaded_model = spacy.load("model_artifacts")     
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )
              
def evaluate_model(
    toeknizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (toeknizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8 # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if predicted_label == 'negative':
                continue
            if score >= 0.5 and true_label['pos']:
                true_positives += 1
            elif score >= 0.5 and true_label['neg']:
                false_positives += 1
            elif score < 0.5 and true_label['neg']:
                true_negatives += 1
            elif score < 0.5 and true_label['pos']:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f-score': f_score}

if __name__ == '__main__':
    # Load data
    dataset = load_dataset("tweet_eval", "sentiment")
    train_data, test_data = dataset['train'].shuffle(seed=42), dataset['test']
    train_model(train_data, test_data)
    
    # Test model
    test_model()
    
