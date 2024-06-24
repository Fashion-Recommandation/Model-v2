import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import re
import os
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import shutil
from django.conf import settings
import traceback



model_path = 'vgg16-397923af.pth'  # Update this path to your local model file
vgg16 = models.vgg16()
vgg16.load_state_dict(torch.load(model_path))
vgg = nn.Sequential(*list(vgg16.children())[:-1])
vgg.eval()

preprocess = transforms.Compose([
     transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img[:3, :, :] if img.shape[0] == 4 else img),  # Ensure 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function to extract features
def extract_features(image_path):
    img = process_image(image_path)
    with torch.no_grad():
        features = vgg(img)
    return features.squeeze().numpy()

def image_to_vector(image_path):
  try:
    for line in traceback.format_stack():
        #print(line.strip())
        pass
    features = extract_features(image_path)
    print(f"Processing image: {image_path}, Shape after preprocessing: {features.shape}")
    return features
  except (OSError, IOError) as e:
        print(f"Error opening {image_path}: {e}")
        return None
  
Up_directory = './core/training/NoBG/Up/'
Down_directory = './core/training/NoBG/Down/'



up_files_dict = {}
up = []
up_files_name = []
for i in range(2, 101):
    up_files = [f for f in os.listdir(Up_directory) if re.match(f'^{i}\..*$', f)]
    for file in up_files:
        up_files_name.append(file)
        x = image_to_vector(f'{Up_directory}{file}')
        up.append(x)
        up_files_dict[file] = x

# Process 'Down' images
down = []
files = os.listdir(Down_directory)
print(files)

one_letter_filenames = [fname for fname in files if len(fname) == 5 and fname[0].isalpha()]
sorted_images = sorted(one_letter_filenames, key=lambda x: x[:2])
sorted_images.append('Z.jpeg')
print(len(sorted_images))

two_letter_a_filenames = [fname for fname in files if len(fname) == 6 and fname[0] == 'A']
sorted_images_a = sorted(two_letter_a_filenames, key=lambda x: x[:2])
sorted_images_a.append('AH.jpeg')
print(sorted_images_a)

two_letter_b_filenames = [fname for fname in files if len(fname) == 6 and fname[0] == 'B']
sorted_images_b = sorted(two_letter_b_filenames, key=lambda x: x[:2])
sorted_images_b.append('BC.jpeg')
print(sorted_images_b)

sorted_filenames = sorted_images + sorted_images_a + sorted_images_b
print(sorted_filenames)
print(len(sorted_filenames))

down_files_dict = {}
for file in sorted_filenames:
    path = f'{Down_directory}{file}'
    x = image_to_vector(path)
    down.append(x)
    down_files_dict[path] = x

label = pd.read_excel('./core/training/Matching Style.xlsx', usecols=lambda x: '0' not in x,)
# print(label.head())

up = np.array(up)
down = np.array(down)
print("Shape of up:", up.shape)
print("Shape of down:", down.shape)



num_rows_down = len(down)
up_resized = up[:num_rows_down]

up_flattened = [u.flatten() for u in up_resized]
down_flattened = [d.flatten() for d in down]

X_combined = []
for u in up_flattened:
    for d in down_flattened:
        X_combined.append(np.concatenate((u, d)))

X_combined = np.array(X_combined)
print("Combined shape before reshaping:", X_combined.shape)


print("Expected size:", 6633 * 512)
print("Actual size:", X_combined.size)

if X_combined.size == 6633 * 512:
    X_combined = X_combined.reshape(6633, 512)
    print("Combined shape after reshaping:", X_combined.shape)
else:
    print("Error: Size mismatch. Cannot reshape array to the desired shape.")

y_score = np.array(label)
print(y_score.shape)
y_score = y_score.reshape((6633))
print(y_score.shape)

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

y_score_float = np.vectorize(safe_float)(y_score).astype(np.float32)



y_score_float_truncated = y_score_float[:len(X_combined)]

X_train, X_test, y_train, y_test = train_test_split(X_combined, y_score_float_truncated, random_state=20, train_size=0.8, shuffle=True)

print("Training set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)


###MODEL###


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("After device")
# Splitting the training data into training and validation sets
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print("After tensor")
X_train, X_val, y_train, y_val = train_test_split(X_train_tensor, y_train_tensor, test_size=0.15, random_state=42, shuffle=True)
print("After train test split")
batch_size = 218

train_dataset = TensorDataset(X_train, y_train)
print("After TensorDataset")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
print("After DataLoader")
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

print("After tensor before ComplexLinearRegression works ")

class ComplexLinearRegression(nn.Module):
    def __init__(self, input_size):
        super(ComplexLinearRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = validate_model(model, val_loader, criterion)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    return val_loss / len(val_loader)

input_size = X_train.shape[1]

# model = ComplexLinearRegression(input_size).to(device)

print("before loading model works ")

criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# 60 -> best epochs number found for VGG16
num_epochs = 60
# best_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
best_model = ComplexLinearRegression(input_size).to(device)
# best_model.load_state_dict(torch.load("./core/fashion_recommedation2.pt"))
best_model.load_state_dict(torch.load("./core/fashion_recommedation2.pt"))

# best_model = torch.load('./core/fashion_recommedation.pt')
best_model.eval()

print("after loading model works ")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()
            predictions.extend(outputs.squeeze().tolist())

    test_loss /= len(test_loader)
    return test_loss, predictions

test_loss, test_predictions = evaluate_model(best_model, test_loader, criterion)

test_predictions = np.array(test_predictions)

print("after evaluate model works ")

print("Predicted\tTrue")
for i in range(len(test_predictions)):
    print(f'{test_predictions[i]:.4f}\t\t{y_test_tensor.cpu().numpy()[i]:.4f}')

print(f'Mean Squared Error on Test Set: {test_loss:.4f}')

one = 0
two = 0
more = 0

for i in range(len(test_predictions)):
  if (abs(test_predictions[i] - y_test[i]) <= 1):
    one += 1
  elif abs(test_predictions[i] - y_test[i]) > 2:
    more += 1
  else:
    two += 1

print(f'one error: {one * 100/len(test_predictions):.3f}%')
print(f'two error: {two * 100/len(test_predictions):.3f}%')
print(f'more error: {more * 100/len(test_predictions):.3f}%')

from rembg import remove
from PIL import Image

def removeBG(input_path):
  output_path = './input_file'
  with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)
  return output_path

def find_threshold(array, th):
  n = torch.sum(array > th).item()
  if th == 5:
    return th, n
  if n < 5:
    return find_threshold(array, th - 1)
  else:
    return th, n
  
import torch
import numpy as np

suggestion_down_directory = './core/training/Up/'
suggestion_up_directory = './core/training/Down/'

def suggesting_process(array, input_vector, clothing_type):
    X = []
    concatenated_vector = []
    
    for arr in array:

      if clothing_type == 'up':
        concatenated_vector.append(np.concatenate((input_vector.flatten(), arr.flatten()))) 
      else:
        concatenated_vector.append(np.concatenate((arr.flatten(), input_vector.flatten())))

    X = np.array(concatenated_vector)

    print("X:", X, '\nX.shape', X.shape)
    return X

def return_model(X, model):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tensor = X_tensor.to(device)
    # model = model.to(device)
    suggest_outputs = model(X_tensor)
    return suggest_outputs

def create_final_suggestions(index_suggestions, suggest_outputs, type):
  final_suggestions = []

  # for i in index_suggestions:
  #   if type == 'up':
  #     name = f'{suggestion_up_directory}{sorted_filenames[i]}'
  #     final_suggestions.append(name)
  #   else:
  #     final_suggestions.append(f'{suggestion_down_directory}{up_files_name[i]}')

  for i in index_suggestions:
        if type == 'up':
            name = f'{suggestion_up_directory}{sorted_filenames[i]}'
        else:
            name = f'{suggestion_down_directory}{up_files_name[i]}'

        # Generate unique filename
        base_filename = os.path.basename(name)
        unique_filename = f'{type}_{i}_{base_filename}'
        destination_path = os.path.join(settings.RECOMMENDATIONS_DIR, unique_filename)

        # Copy image to the recommendations directory
        shutil.copy(name, destination_path)

        # Construct the URL for the saved image
        url = f'/recommendations/{unique_filename}'
        score = suggest_outputs[i]

        # Add the URL to final suggestions
        final_suggestions.append({'image':url, 'score': score})
  return final_suggestions


def find_suggestions(input_file, clothing_type):
    # In this function we need an up file to suggest the down side for it
    input_vector = image_to_vector(input_file)

    if clothing_type == 'up':
        X = suggesting_process(down, input_vector, clothing_type)
    elif clothing_type == 'down':
        X = suggesting_process(up, input_vector, clothing_type)

    suggest_outputs = return_model(X, best_model)
    threshold, number = find_threshold(suggest_outputs, 9)
    print("threshold:", threshold)

    index_suggestions = []
    count = 0
    for sug in suggest_outputs:
        if sug[0] > threshold:
            index_suggestions.append(count)
        count += 1

    print("index suggestions length:", len(index_suggestions))
    final_suggestions = create_final_suggestions(index_suggestions, suggest_outputs, clothing_type)
    print("final_suggestions", final_suggestions)
    return final_suggestions

# test = './core/94.png'
# outputs = find_suggestions(test, 'up')
# print(outputs)

