import pandas as pd 
import numpy as np
from tqdm.auto import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer ,AutoTokenizer
import torch.nn as nn 
from torch.utils.data  import Dataset
from torch.utils.data import Dataset, DataLoader , TensorDataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class SubjectDataset(Dataset):
    def __init__(self , df):
        self.df=df
        self.maxlen=256
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    def __len__(self):
        return len(self.df)
    def __getitem__(self , index):
        sample_title= str(self.df['title'].iloc[index])
        sample_content= str(self.df['text'].iloc[index])
        sample_subject= str(self.df['subject'].iloc[index])
        sample = sample_title + " " + sample_content + " " + sample_subject

                # SBERT returns embeddings directly
        embedding = self.model.encode(sample, convert_to_tensor=True)
        label = torch.tensor(self.df['label'].iloc[index], dtype=torch.long)
        
        return {
            'embedding': embedding,
            'labels': label
        }
def get_sbert_embeddings(dataset, sbert_model, device):
    embeddings = [] # Will store SBERT embeddings for each text
    labels = [] ## for labels
    sbert_model = sbert_model.to(device)
    sbert_model.eval()
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting SBERT embeddings"):
            batch_embeddings = batch['embedding'].cpu().numpy()
            embeddings.extend(batch_embeddings)
            labels.extend(batch['labels'].cpu().numpy())
    # return np.array(embeddings), np.array(labels)
    return embeddings, labels

def generate_sbert_embeddings(df_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    # Chargement des données
    df = pd.read_csv(df_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split des données
    df_train, df_test = train_test_split(df, train_size=0.8, random_state=42)
    
    # Chargement du modèle BERT
    sbert_model = SentenceTransformer(model_name)
    
    # Génération des embeddings
    print("Génération des embeddings pour l'ensemble d'entraînement")
    X_train, y_train = get_sbert_embeddings(SubjectDataset(df_train), sbert_model, device)
    
    print("Génération des embeddings pour l'ensemble de test")
    X_test, y_test = get_sbert_embeddings(SubjectDataset(df_test), sbert_model, device)
    
    # Sauvegarde des embeddings
    np.save('X_train_sbert_embeddings.npy', X_train)
    np.save('y_train_sbert_labels.npy', y_train)
    np.save('X_test_sbert_embeddings.npy', X_test)
    np.save('y_test_sbert_labels.npy', y_test)

    # Sauvegarder le modèle BERT
    sbert_model.save('sbert_model')
    return X_train, y_train, X_test, y_test ,sbert_model

class SBERTClassifier(nn.Module):
    def __init__(self, embedding_dim=384, num_classes=2, dropout_rate=0.2):
        super(SBERTClassifier, self).__init__()
                
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, embeddings):
        return self.classifier(embeddings)
    
def train_Sbert_classifier(X_train, y_train, X_test, y_test, num_classes=2, batch_size=32, epochs=10):    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = SBERTClassifier(
        embedding_dim=X_train.shape[1], 
        num_classes=num_classes, 
        dropout_rate=0.2
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_accuracy = 0
    best_model_state = None
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            _, y_pred = torch.max(test_outputs, 1)
            y_pred = y_pred.cpu().numpy()
        
        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Save metrics
        history['val_accuracy'].append(accuracy)
        history['val_f1'].append(f1)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
    
    print(f"\nBest Accuracy: {best_accuracy:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Compute full classification report
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor.to(device)).argmax(dim=1).cpu().numpy()
    
    print("\nPerformance on test set:")
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_train.shape[1],
        'num_classes': num_classes,
        'dropout_rate': 0.2
    }, 'sbert_classifier.pth')
    plot_training_metrics(history, epochs)
    return model


def plot_training_metrics(history, epochs):
    """Plot training and validation metrics."""
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'o-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history['val_accuracy'], 'o-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history['val_f1'], 'o-', label='Validation F1 Score')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot Precision and Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history['val_precision'], 'o-', label='Validation Precision')
    plt.plot(epochs_range, history['val_recall'], 'o-', label='Validation Recall')
    plt.title('Validation Precision and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_evaluation.png')
    plt.show()



def main():
    # Chemin vers votre fichier CSV
    df_path = "/kaggle/input/the-liar-dataset/data.csv"
    
    # Générer les embeddings et sauvegarder les modèles
    X_train, y_train, X_test, y_test, sbert_model = generate_sbert_embeddings(df_path)
    
    # Entraîner le classificateur CNN
    SBERT_model = train_Sbert_classifier(X_train, y_train, X_test, y_test ,2 )
    
    return SBERT_model

if __name__ == "__main__":
    sbert_model = main()
