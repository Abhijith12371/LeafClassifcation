
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# =======================
# CONFIGURATION
# =======================
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced for better gradient updates
EPOCHS = 50  # Increased
LEARNING_RATE = 1e-4

BASE_DIR = "plant_dataset"
# Use final_dataset with proper train/val split
TRAIN_DIR = os.path.join(BASE_DIR, "final_dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "final_dataset", "val")
MODEL_SAVE_PATH = "plant_disease_model_final.keras"

def build_model(num_classes):
    """
    Enhanced MobileNetV2 model with BatchNormalization
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Enhanced custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def train():
    # Verify directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: Training directory not found at {TRAIN_DIR}")
        print("   Please run the dataset creation script first!")
        return
    
    if not os.path.exists(VAL_DIR):
        print(f"❌ Error: Validation directory not found at {VAL_DIR}")
        print("   Please run the dataset creation script first!")
        return

    # =======================
    # DATA GENERATORS (FIXED)
    # =======================
    print("🎨 Setting up data generators...")
    
    # Training: Moderate augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Added for plant leaves
        fill_mode="nearest"
    )

    # Validation: NO augmentation, only preprocessing
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print(f"📦 Loading training data from {TRAIN_DIR}...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    print(f"📦 Loading validation data from {VAL_DIR}...")
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_generator.num_classes
    print(f"🔍 Found {num_classes} classes.")
    print(f"📊 Training samples: {train_generator.samples}")
    print(f"📊 Validation samples: {val_generator.samples}")

    # Check class distribution
    print("\n📋 Class distribution:")
    for class_name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
        print(f"   {idx}: {class_name}")

    # =======================
    # BUILD MODEL
    # =======================
    model = build_model(num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()

    # =======================
    # CALLBACKS
    # =======================
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # =======================
    # PHASE 1: TRAIN HEAD
    # =======================
    print("\n" + "="*50)
    print("🚀 PHASE 1: Training Classification Head")
    print("="*50)
    
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 1 complete.")
    print(f"   Best val_accuracy: {max(history1.history['val_accuracy']):.4f}")

    # =======================
    # PHASE 2: FINE-TUNING
    # =======================
    print("\n" + "="*50)
    print("🔓 PHASE 2: Fine-tuning MobileNetV2")
    print("="*50)
    
    # Unfreeze the whole model first
    model.trainable = True
    
    # We added 7 layers in build_model:
    # GAP, BN, Dropout, Dense, BN, Dropout, Output
    head_layers_count = 7
    base_layers_count = len(model.layers) - head_layers_count
    
    # We want to fine-tune the top 50 layers of the BASE model
    fine_tune_start = base_layers_count - 50
    if fine_tune_start < 0: fine_tune_start = 0
    
    print(f"   Total layers: {len(model.layers)}")
    print(f"   Base model layers: {base_layers_count}")
    print(f"   Fine-tuning starts at layer: {fine_tune_start}")
    print("   (Keeping BatchNormalization layers FROZEN in base)")
    
    # Iterate through the base layers
    for i, layer in enumerate(model.layers[:base_layers_count]):
        # Freeze if:
        # 1. It is below our fine-tune cutoff
        # 2. OR it is a BatchNormalization layer (crucial for MobileNetV2)
        if i < fine_tune_start or isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            
    # The head layers (last 7) remain trainable (from model.trainable=True)

    trainable_count = len([l for l in model.layers if l.trainable])
    print(f"   Trainable layers: {trainable_count}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Update callbacks for phase 2
    callbacks[1] = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    history2 = model.fit(
        train_generator,
        initial_epoch=history1.epoch[-1],
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # =======================
    # SAVE FINAL MODEL
    # =======================
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Training complete!")
    print(f"   Model saved to: {MODEL_SAVE_PATH}")
    print(f"   Final val_accuracy: {max(history2.history['val_accuracy']):.4f}")

    # Save class indices
    with open("class_indices.json", "w") as f:
        json.dump(train_generator.class_indices, f, indent=2)
    print(f"   Class mapping saved to: class_indices.json")

if __name__ == "__main__":
    train()
