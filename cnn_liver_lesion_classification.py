import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

# =========================================================
# 1. DATASET
# =========================================================

data_dir = "/content/liver_data"
img_size = (224, 224)
batch_size = 16
seed = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

val_batches = tf.data.experimental.cardinality(temp_ds).numpy()
val_ds = temp_ds.take(val_batches // 2)
test_ds = temp_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("Classes:", class_names)

# =========================================================
# 2. CLASS WEIGHTS
# =========================================================

y_train = []
for _, y in train_ds.unbatch():
    y_train.append(y.numpy())

y_train = np.array(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# =========================================================
# 3. MODELS (5 MODELS)
# =========================================================

models_dict = {
    "ResNet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input
    ),
    "MobileNetV2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input
    ),
    "DenseNet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input
    ),
    "EfficientNetB3": (
        tf.keras.applications.EfficientNetB3,
        tf.keras.applications.efficientnet.preprocess_input
    ),
    "VGG19": (
        tf.keras.applications.VGG19,
        tf.keras.applications.vgg19.preprocess_input
    )
}

results = {}
history_dict = {}

# =========================================================
# 4. MODEL BUILDER
# =========================================================

def build_model(model_fn, preprocess_fn):

    base_model = model_fn(
        include_top=False,
        weights="imagenet",
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224,224,3))

    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    x = tf.keras.layers.Lambda(preprocess_fn)(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model

# =========================================================
# 5. TRAINING LOOP
# =========================================================

for name, (model_fn, prep_fn) in models_dict.items():

    print("\n==============================")
    print("TRAINING:", name)
    print("==============================")

    tf.keras.backend.clear_session()

    model, base_model = build_model(model_fn, prep_fn)

    # --------------------------
    # TRAIN
    # --------------------------
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        class_weight=class_weights,
        verbose=1
    )

    # --------------------------
    # FINE TUNING
    # --------------------------
    base_model.trainable = True

    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        class_weight=class_weights,
        verbose=1
    )

    history_dict[name] = (h1, h2)

    # =====================================================
    # EVALUATION
    # =====================================================

    y_true, y_pred, y_prob = [], [], []

    for x, y in test_ds:
        preds = model.predict(x)

        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = np.mean(y_true == y_pred)
    results[name] = acc

    print("\nTEST ACCURACY:", acc)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(name + " Confusion Matrix")
    plt.show()

    # =====================================================
    # ROC-AUC
    # =====================================================

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(7,6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} AUC={roc_auc:.3f}")

    plt.plot([0,1], [0,1], "k--")
    plt.title(name + " ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

# =========================================================
# 6. ACCURACY + LOSS PLOTS
# =========================================================

def plot_history(h1, h2, name):

    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(14,5))
    plt.suptitle(name)

    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.grid()

    plt.show()

for name in history_dict:
    h1, h2 = history_dict[name]
    plot_history(h1, h2, name)

# =========================================================
# 7. MODEL COMPARISON (COLORED BAR CHART)
# =========================================================

df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
df = df.sort_values(by="Accuracy", ascending=False)

colors = ["red", "blue", "green", "orange", "purple"]

plt.figure(figsize=(9,5))

bars = plt.bar(df["Model"], df["Accuracy"], color=colors)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.3f}", ha='center')

plt.title("5-Model Comparison (Liver Disease Classification)")
plt.ylim(0,1)
plt.grid(axis='y')
plt.show()
