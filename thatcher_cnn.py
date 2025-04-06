import cv2
import dlib
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torchvision.models import VGG16_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import pandas as pd
from collections import defaultdict


# Set device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# Load model
print("Loading VGG16 model...")
vgg16_model = models.vgg16(weights=VGG16_Weights.DEFAULT).to(device)

vgg16_model.eval()

# Define feature extractor
feature_extractor = nn.Sequential(
    vgg16_model.features,
    nn.AdaptiveAvgPool2d((7, 7)),
    nn.Flatten(),
    *list(vgg16_model.classifier.children())[:5]
).to(device)

# Preprocessing transform for images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Thatcherization process
def thatcherize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(gray)
    if not faces:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
    img = image.copy()

    for indices in [(36, 42), (42, 48), (48, 68)]:
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(*indices)]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        region = img[y_min:y_max, x_min:x_max]
        img[y_min:y_max, x_min:x_max] = cv2.flip(region, 0)

    return img



# Feature extraction
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    if 'inverted' in image_path:
        img = cv2.rotate(img, cv2.ROTATE_180)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(gray)
    if not faces:
        return None
    face = faces[0]
    landmarks = predictor(gray, face)
    
    # Facial geometry features
    left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
    right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
    mouth_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)], axis=0)
    eye_distance = np.linalg.norm(left_eye - right_eye)
    eye_mouth_distance = np.linalg.norm((left_eye + right_eye) / 2 - mouth_center)
    mouth_width = np.linalg.norm(
        np.array([landmarks.part(48).x, landmarks.part(48).y]) -
        np.array([landmarks.part(54).x, landmarks.part(54).y])
    )
    left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    left_eye_width = np.linalg.norm(left_eye_points[3] - left_eye_points[0])
    left_eye_height = np.linalg.norm(np.mean(left_eye_points[1:3], axis=0) - np.mean(left_eye_points[4:6], axis=0))
    left_eye_aspect_ratio = left_eye_width / (left_eye_height + 1e-6)
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # CNN features
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    # print(f"Input tensor device: {input_tensor.device}")
    with torch.no_grad():
        cnn_features = feature_extractor(input_tensor).cpu().numpy().squeeze()


    # Combine all features
    return np.concatenate([
        [eye_distance, eye_mouth_distance, mouth_width, left_eye_aspect_ratio, eye_angle],
        cnn_features
    ])

# Train & evaluate model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def train_and_evaluate(dataset_path):
    print("Extracting features and training SVM...")
    features, labels, paths = [], [], []
    categories = {
        'normal_upright': 0,
        'thatcherized_upright': 1,
        'normal_inverted': 0,
        'thatcherized_inverted': 1
    }

    label_names = ['Normal', 'Thatcherized']
    
    for category, label in categories.items():
        category_path = os.path.join(dataset_path, category)
        for img_name in tqdm(os.listdir(category_path), desc=f"Extracting {category}"):
            img_path = os.path.join(category_path, img_name)
            feat = extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(label)
                paths.append(img_path)

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(
        X, y, paths, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.001, 0.01],
        'kernel': ['rbf']
    }

    grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    svm = grid_search.best_estimator_
    y_pred = svm.predict(X_test)



    # Store misclassified examples
    false_positives = []
    false_negatives = []

    # Go through each prediction and compare
    for true_label, pred_label, path in zip(y_test, y_pred, test_paths):
        if pred_label == 1 and true_label == 0:
            false_positives.append(path)
        elif pred_label == 0 and true_label == 1:
            false_negatives.append(path)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc:.2f}")

    #Per-orientation accuracy
    for cat in ['upright', 'inverted']:
        mask = [cat in p for p in test_paths]
        if any(mask):
            cat_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"   {cat.capitalize()} Accuracy: {cat_acc:.2f}")

    #Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


    # Labels
    label_names = ['Normal', 'Thatcherized']


    #Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Show false positives and false negatives
    show_images(false_positives, "False Positive")
    show_images(false_negatives, "False Negative")

    # Plot Classification Report as Heatmap
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8, 4))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu')
    plt.title("Classification Report Heatmap")
    plt.tight_layout()
    plt.savefig("classification_report_heatmap.png")
    plt.show()

    # ROC Curve
    y_score = svm.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()

    #Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    plt.show()

# Show only first 5 to avoid overload
def show_images(image_paths, title):
    for path in image_paths[:5]:  
        img = cv2.imread(path)
        if img is not None:
            cv2.imshow(title, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Entry point
if __name__ == "__main__":
    train_and_evaluate('out')
