# ============================================================
# Hautkrankheiten-Klassifikation mit KI (HAM10000 Datensatz)
# Jugend forscht Projekt - Bilderkennung mit Transfer Learning
# ============================================================
#
# ANLEITUNG:
# 1. Öffnet Google Colab: https://colab.research.google.com
# 2. Laufzeit > Laufzeittyp ändern > GPU auswählen
# 3. Fügt diese Zellen nacheinander in Colab ein und führt sie aus
# 4. Bevor ihr startet: Ladet euren Kaggle API-Key (kaggle.json) hoch
#    (Kaggle-Konto > Account > "Create New API Token")
# ============================================================


# ------------------------------------------------------------
# ZELLE 1: Kaggle-Zugang einrichten und Datensatz herunterladen
# ------------------------------------------------------------
"""
from google.colab import files
files.upload()  # hier eure kaggle.json hochladen

import os
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 600)

!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
!unzip -q skin-cancer-mnist-ham10000.zip -d ham10000_data
"""


# ------------------------------------------------------------
# ZELLE 2: Bibliotheken importieren
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

print("TensorFlow Version:", tf.__version__)
print("GPU verfügbar:", tf.config.list_physical_devices('GPU'))


# ------------------------------------------------------------
# ZELLE 3: Metadaten laden und Klassen verstehen
# ------------------------------------------------------------
# Pfad ggf. anpassen, je nachdem wie der Datensatz entpackt wurde
DATA_DIR = 'ham10000_data'
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')

df = pd.read_csv(METADATA_PATH)

# Die 7 Diagnose-Kategorien in HAM10000
label_map = {
    'akiec': 'Aktinische Keratose',
    'bcc': 'Basalzellkarzinom',
    'bkl': 'Benigne Keratose',
    'df': 'Dermatofibrom',
    'mel': 'Melanom',
    'nv': 'Melanozytischer Nävus',
    'vasc': 'Vaskuläre Läsion'
}

df['label_name'] = df['dx'].map(label_map)
print(df['label_name'].value_counts())

# WICHTIG: Der Datensatz ist stark unausgeglichen!
# 'nv' (gutartige Muttermale) macht ca. 67% aus - das müsst ihr
# in eurer Auswertung unbedingt thematisieren (siehe Zelle 8)
df['label_name'].value_counts().plot(kind='bar', figsize=(10, 5))
plt.title('Verteilung der Diagnosen im Datensatz')
plt.ylabel('Anzahl Bilder')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# ZELLE 4: Bildpfade zuordnen und Daten aufteilen
# ------------------------------------------------------------
# Bilder liegen in zwei Ordnern (HAM10000_images_part_1 und _2)
def find_image_path(image_id):
    for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        path = os.path.join(DATA_DIR, part, image_id + '.jpg')
        if os.path.exists(path):
            return path
    return None

df['path'] = df['image_id'].apply(find_image_path)
df = df.dropna(subset=['path'])  # falls Pfade fehlen

# Aufteilen: 70% Training, 15% Validierung, 15% Test
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['dx'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=42
)

print(f"Training: {len(train_df)} Bilder")
print(f"Validierung: {len(val_df)} Bilder")
print(f"Test: {len(test_df)} Bilder")


# ------------------------------------------------------------
# ZELLE 5: Datenaugmentation und Generatoren
# ------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

# Validierung und Test NUR normalisieren, keine Augmentation!
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='path', y_col='dx',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df, x_col='path', y_col='dx',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df, x_col='path', y_col='dx',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

NUM_CLASSES = len(train_generator.class_indices)
print("Klassen:", train_generator.class_indices)


# ------------------------------------------------------------
# ZELLE 6: Transfer-Learning-Modell aufbauen (MobileNetV2)
# ------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,       # eigene Klassifikationsschicht statt Original
    weights='imagenet'
)
base_model.trainable = False  # vortrainierte Gewichte einfrieren

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)          # gegen Overfitting
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ------------------------------------------------------------
# ZELLE 7: Modell trainieren
# ------------------------------------------------------------
# Da der Datensatz unausgeglichen ist, Klassengewichte berechnen
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['dx']),
    y=train_df['dx']
)
class_weight_dict = dict(zip(np.unique(train_df['dx']).argsort(), class_weights))

EPOCHS = 15

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weight_dict
)

# Trainingsverlauf visualisieren
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validierung')
ax1.set_title('Genauigkeit über Epochen')
ax1.set_xlabel('Epoche')
ax1.legend()

ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validierung')
ax2.set_title('Loss über Epochen')
ax2.set_xlabel('Epoche')
ax2.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# ZELLE 8: Modell auswerten - Confusion Matrix & Klassenmetriken
# ------------------------------------------------------------
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# Detaillierter Bericht: Precision, Recall, F1 pro Klasse
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix als Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Vorhergesagt')
plt.ylabel('Tatsächlich')
plt.title('Confusion Matrix - welche Krankheiten werden verwechselt?')
plt.tight_layout()
plt.show()

# WICHTIG FÜR EURE DOKUMENTATION:
# Schaut euch an, WELCHE Klassen oft verwechselt werden.
# Häufiges Muster: Melanom (mel) wird mit Nävus (nv) verwechselt,
# weil beide visuell ähnlich sein können - das ist ein spannender
# Diskussionspunkt für eure Arbeit!


# ------------------------------------------------------------
# ZELLE 9 (BONUS): Modell speichern und einzelnes Bild testen
# ------------------------------------------------------------
model.save('hautkrankheiten_modell.h5')

def predict_single_image(image_path, model, class_names):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    print(f"Vorhersage: {label_map.get(predicted_class, predicted_class)}")
    print(f"Konfidenz: {confidence:.1f}%")

    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()

# Beispiel-Aufruf:
# predict_single_image('pfad/zu/testbild.jpg', model, class_names)
