# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

file_path = './data/spooky_author_data.csv'
def load_and_prepare_data(filepath=file_path, test_size=0.2, random_state=42):
    """Load dataset and split into train/test sets with stratification."""
    df = pd.read_csv(filepath)
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['author']
    )

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Generate sentence embeddings using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(texts)} texts...")
    return model.encode(texts, show_progress_bar=True)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Visualize confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def train_and_evaluate(X_train, X_test, y_train, y_test, class_names):
    """Train logistic regression and evaluate performance."""
    # Initialize and train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Generate evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)

    return clf

def main():
    # Step 1: Load and split data
    train_df, test_df = load_and_prepare_data()

    # Step 2: Generate embeddings
    X_train = generate_embeddings(train_df['text'].tolist())
    X_test = generate_embeddings(test_df['text'].tolist())

    # Step 3: Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['author'])
    y_test = le.transform(test_df['author'])

    # Step 4: Train and evaluate model
    _ = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_)

if __name__ == "__main__":
    main()
