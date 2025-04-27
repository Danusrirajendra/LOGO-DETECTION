from tkinter import *
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import pickle
import requests
from io import BytesIO
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc)
from itertools import cycle # Needed for ROC curve plotting colors

# --- Attempt to import plotting libraries ---
PLOT_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    print("Warning: Matplotlib or Seaborn not found. Plots will not be generated.")

# Initialize main window
main = Tk()
main.title("Fake Logo Detection")
main.geometry("1300x1200")

# --- Global Variables ---
filename = ""
labels = []
X, Y = None, None
X_train, y_train = None, None
X_test, y_test = None, None
classifier = None
resnet_model = None
output_text = None
bg_label = None

# --- Custom Fonts ---
title_font = ("Helvetica", 30, "bold")
button_font = ("Helvetica", 12, "bold")
text_font = ("Helvetica", 10)

# --- Background Image Function ---
def load_background_from_url(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        bg_image = Image.open(image_data)
        # Use Image.Resampling.LANCZOS if available (Pillow >= 9.1.0)
        try:
            bg_image = bg_image.resize((1300, 1200), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback for older Pillow versions
            bg_image = bg_image.resize((1300, 1200), Image.LANCZOS)
        return ImageTk.PhotoImage(bg_image)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching background image URL: {e}")
        return None
    except Exception as e:
        print(f"Error processing background image: {e}")
        return None

# --- Set Background ---
image_url = "https://wallpapercave.com/wp/wp11155228.jpg" # Example URL
bg_photo = load_background_from_url(image_url)
if bg_photo:
    bg_label = Label(main, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
else:
    main.configure(bg="#DDDDDD") # Fallback background color

# --- Navigation Functions ---
def change_page(page_func):
    # Destroy all widgets except the background label
    active_widgets = main.winfo_children()
    for widget in active_widgets:
        if bg_label is not None and widget == bg_label:
            continue
        widget.destroy()
    # Call the function to build the new page
    page_func()

def add_navigation_buttons(current_page):
    # Ensure background is always present and behind other widgets
    if bg_label is not None and bg_photo:
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.lower() # Keep background label at the bottom

    # Navigation Button Frames (with white background for visibility)
    back_frame = Frame(main, bg="white", bd=0)
    back_frame.place(relx=0.1, rely=0.05, anchor="center") # Top-left area
    back_button = Button(back_frame, text="Back", font=button_font, bg="white", fg="black", bd=0, padx=20, pady=5, relief="flat", command=lambda: change_page(prev_page(current_page)))
    back_button.pack()

    next_frame = Frame(main, bg="white", bd=0)
    next_frame.place(relx=0.9, rely=0.05, anchor="center") # Top-right area
    next_button = Button(next_frame, text="Next", font=button_font, bg="white", fg="black", bd=0, padx=20, pady=5, relief="flat", command=lambda: change_page(next_page(current_page)))
    next_button.pack()

PAGES = ['home_page', 'upload_page', 'process_page', 'train_page', 'result_page']

def prev_page(current_page_func):
    current_name = current_page_func.__name__
    try:
        index = PAGES.index(current_name)
        if index > 0:
            return globals()[PAGES[index - 1]]
        return home_page # Stay on home if already first
    except ValueError:
        return home_page # Default to home if name not found

def next_page(current_page_func):
    current_name = current_page_func.__name__
    try:
        index = PAGES.index(current_name)
        if index < len(PAGES) - 1:
            return globals()[PAGES[index + 1]]
        return result_page # Stay on result if already last
    except ValueError:
        return result_page # Default to result if name not found


# --- Page Definitions ---
def home_page():
    global output_text
    add_navigation_buttons(home_page) # Pass the function itself

    # Title Frame (centered)
    title_frame = Frame(main, bg="white", bd=0)
    title_frame.place(relx=0.5, rely=0.12, anchor="center") # Adjusted rely for better spacing
    title = Label(title_frame, text="WELCOME TO LOGO IDENTIFICATION", font=title_font, bg="white", fg="pink") # Example colors
    title.pack(padx=10, pady=5)

    # Description Frame (centered, larger)
    description_frame = Frame(main, bg="white", highlightbackground="pink", highlightthickness=5)
    description_frame.place(relx=0.5, rely=0.55, anchor="center", width=1000, height=600) # Increased size

    description_text = """
Welcome to the Fake Logo Detection application!

This tool utilizes deep learning models to help distinguish between genuine and potentially counterfeit logos in images.

How it works:
1.  Upload: Select a folder containing your dataset. The folder should have subfolders for each logo category (e.g., 'Original_Nike', 'Fake_Nike', 'Original_Adidas', 'Fake_Adidas').
2.  Process: The application will automatically preprocess the images (resize, normalize) and split them into training and testing sets.
3.  Train: Train two different models - a custom Convolutional Neural Network (CNN) and a pre-trained ResNet50 model (using transfer learning) - on your data. Evaluation metrics and graphs (like confusion matrices) will be generated.
4.  Classify: Upload a single logo image, and the trained models will predict its category (e.g., 'Original_Nike' or 'Fake_Adidas').

Navigate using the 'Next' and 'Back' buttons. Start by selecting your dataset folder on the 'Upload' page.
"""
    description = Label(description_frame, text=description_text, font=("Arial", 15), bg="white", fg="black", wraplength=950, justify="left")
    description.pack(padx=20, pady=20, fill="both", expand=True)

def upload_page():
    global output_text
    add_navigation_buttons(upload_page)

    # Title
    title_frame = Frame(main, bg="white", bd=0)
    title_frame.place(relx=0.5, rely=0.15, anchor="center")
    title = Label(title_frame, text="Upload Dataset", font=title_font, bg="white", fg="#333333")
    title.pack(padx=10, pady=5)

    # Upload Button
    upload_frame = Frame(main, bg="#4CAF50", bd=0) # Green background for button frame
    upload_frame.place(relx=0.5, rely=0.3, anchor="center")
    upload_button = Button(upload_frame, text="Select Dataset Folder", font=button_font, bg="#4CAF50", fg="white", bd=0, padx=20, pady=10, relief="flat", command=uploadDataset)
    upload_button.pack()

    # Text Area for Output
    text_frame = Frame(main, bd=1, relief="solid") # Added border for clarity
    text_frame.place(relx=0.5, rely=0.65, relwidth=0.8, relheight=0.5, anchor="center") # Centered, takes large portion
    output_text = Text(text_frame, font=text_font, bg="white", fg="#333333", wrap=WORD, bd=0)
    output_text.pack(padx=1, pady=1, fill="both", expand=True)

    # Initial text display
    if filename:
        output_text.insert(END, f"Currently selected dataset folder:\n{filename}\n\n")
        img_count = count_images(filename)
        output_text.insert(END, f"Total images found in subfolders: {img_count}\n\n")
        if labels:
            output_text.insert(END, "Detected categories:\n")
            for lbl in labels:
                output_text.insert(END, f"- {lbl}\n")
        else:
             readLabels(filename) # Try reading labels if filename is set but labels aren't
             if labels:
                 output_text.insert(END, "Detected categories:\n")
                 for lbl in labels:
                     output_text.insert(END, f"- {lbl}\n")
             else:
                output_text.insert(END, "\nNo categories (subfolders) detected yet.\n")
    else:
        output_text.insert(END, "Please select the main folder containing your logo dataset.\n")
        output_text.insert(END, "This folder should contain subfolders, where each subfolder represents a category (e.g., 'original_logo', 'fake_logo').\n")

def uploadDataset():
    global filename, labels, X, Y, X_train, y_train, X_test, y_test, classifier, resnet_model
    selected_dir = filedialog.askdirectory(initialdir=".", title="Select Dataset Folder")
    if selected_dir:
        filename = selected_dir
        # Reset everything when a new dataset is selected
        labels = []
        X, Y = None, None
        X_train, y_train = None, None
        X_test, y_test = None, None
        classifier = None
        resnet_model = None

        # Clean up previously saved model/data files
        model_dir = "model"
        files_to_delete = [
            os.path.join(model_dir, "X.npy"),
            os.path.join(model_dir, "Y.npy"),
            os.path.join(model_dir, "labels.pkl"), # Also save/delete labels list
            os.path.join(model_dir, "logo_cnn_best.keras"),
            os.path.join(model_dir, "resnet50_best.keras"),
            os.path.join(model_dir, 'history.pkl'),
            os.path.join(model_dir, 'resnet_history.pkl'),
            os.path.join(model_dir, 'cnn_confusion_matrix.png'),
            os.path.join(model_dir, 'resnet_confusion_matrix.png'),
            os.path.join(model_dir, 'cnn_roc_curve.png'),
            os.path.join(model_dir, 'resnet_roc_curve.png'),
            os.path.join(model_dir, 'model_comparison.png')
        ]

        for f_path in files_to_delete:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    print(f"Deleted old file: {f_path}")
            except OSError as e:
                print(f"Error deleting file {f_path}: {e}")

        # Read labels from the new directory
        readLabels(filename)

        # Update the text area in the upload page
        if output_text:
            output_text.delete(1.0, END)
            output_text.insert(END, f"Dataset folder selected: {filename}\n\n")
            img_count = count_images(filename)
            output_text.insert(END, f"Total images found across subfolders: {img_count}\n\n")
            if labels:
                output_text.insert(END, "Logo categories found (subfolders):\n")
                for label in labels:
                    output_text.insert(END, f"- {label}\n")
                # Save labels list
                if not os.path.exists(model_dir): os.makedirs(model_dir)
                with open(os.path.join(model_dir, 'labels.pkl'), 'wb') as f:
                    pickle.dump(labels, f)
            else:
                output_text.insert(END, "No subfolders found in the selected directory. Each subfolder should represent a category.\n")
        else:
             # If called from elsewhere or text area not yet created (unlikely but safe)
             print(f"Dataset folder selected: {filename}")
             if labels: print(f"Categories: {labels}")


def count_images(directory):
    count = 0
    if not os.path.isdir(directory):
        return 0
    # Count images only within the *immediate* subdirectories (categories)
    for category_folder in os.listdir(directory):
        category_path = os.path.join(directory, category_folder)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                 # Check for common image extensions
                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    count += 1
    return count

def readLabels(directory):
    global labels
    current_labels = []
    if not os.path.isdir(directory):
        labels = []
        return
    # Labels are the names of the subdirectories
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            current_labels.append(item)
    labels = sorted(current_labels) # Sort for consistency


def process_page():
    global output_text, labels
    add_navigation_buttons(process_page)

    # Attempt to load labels if not already loaded
    if not labels:
        model_dir = "model"
        labels_path = os.path.join(model_dir, 'labels.pkl')
        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'rb') as f:
                    labels = pickle.load(f)
            except Exception as e:
                print(f"Error loading labels file: {e}")
                labels = [] # Reset if loading failed

    # Title
    title_frame = Frame(main, bg="white", bd=0)
    title_frame.place(relx=0.5, rely=0.15, anchor="center")
    title = Label(title_frame, text="Process Dataset", font=title_font, bg="white", fg="#333333")
    title.pack(padx=10, pady=5)

    # Process Button
    process_frame = Frame(main, bg="#4CAF50", bd=0)
    process_frame.place(relx=0.5, rely=0.3, anchor="center")
    process_button = Button(process_frame, text="Preprocess Dataset", font=button_font, bg="#4CAF50", fg="white", bd=0, padx=20, pady=10, relief="flat", command=processDataset)
    process_button.pack()

    # Text Area
    text_frame = Frame(main, bd=1, relief="solid")
    text_frame.place(relx=0.5, rely=0.65, relwidth=0.8, relheight=0.5, anchor="center")
    output_text = Text(text_frame, font=text_font, bg="white", fg="#333333", wrap=WORD, bd=0)
    output_text.pack(padx=1, pady=1, fill="both", expand=True)

    # Display status
    if X is not None and Y is not None:
        output_text.insert(END, f"Preprocessed data already generated or loaded for:\n{filename}\n")
        output_text.insert(END, f"Total images processed: {X.shape[0]}\n")
        if X_train is not None:
            output_text.insert(END, f"Train images: {X_train.shape[0]}\n")
            output_text.insert(END, f"Test images: {X_test.shape[0]}\n")
        else:
            output_text.insert(END, "Data loaded, but not yet split into train/test sets.\n")
    elif filename:
        output_text.insert(END, f"Dataset folder selected: {filename}\n")
        if labels:
            output_text.insert(END, f"Categories found: {len(labels)} ({', '.join(labels)})\n")
        else:
            output_text.insert(END, "No categories found. Please ensure the dataset folder has subfolders for categories.\n")
        output_text.insert(END, "\nReady to preprocess. Click the button above.\n")
        output_text.insert(END, "Preprocessing involves resizing images to 64x64, normalizing pixel values, and splitting data.\n")
    else:
        output_text.insert(END, "Please select a dataset folder first on the 'Upload' page.\n")

def processDataset():
    global filename, labels, X, Y, X_train, y_train, X_test, y_test, output_text

    if not filename:
        messagebox.showerror("Error", "No dataset folder selected. Please go back to 'Upload'.")
        return

    # Ensure labels are loaded or read again
    if not labels:
        readLabels(filename)
        if not labels:
             messagebox.showerror("Error", "No categories (subfolders) found in the dataset folder.")
             return
        # Save labels if read successfully
        model_dir = "model"
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        with open(os.path.join(model_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(labels, f)

    if output_text:
        output_text.delete(1.0, END)
        output_text.insert(END, f"Starting dataset processing: {filename}\n")
        output_text.insert(END, f"Found categories: {labels}\n")
        main.update_idletasks() # Update GUI to show message

    X_data, Y_data = [], []
    try:
        model_dir = "model"
        x_path = os.path.join(model_dir, "X.npy")
        y_path = os.path.join(model_dir, "Y.npy")
        labels_path = os.path.join(model_dir, "labels.pkl") # Path for labels

        # Check if processed data and labels exist and match
        labels_match = False
        if os.path.exists(labels_path):
             try:
                 with open(labels_path, 'rb') as f:
                     saved_labels = pickle.load(f)
                 if saved_labels == labels:
                     labels_match = True
                 else:
                    if output_text: output_text.insert(END, "Warning: Saved labels mismatch current labels. Reprocessing needed.\n")
             except Exception as e:
                 if output_text: output_text.insert(END, f"Warning: Error loading saved labels ({e}). Reprocessing needed.\n")


        if os.path.exists(x_path) and os.path.exists(y_path) and labels_match:
            try:
                X_loaded = np.load(x_path)
                Y_indices_loaded = np.load(y_path)
                # Basic validation
                if len(X_loaded) == len(Y_indices_loaded) and Y_indices_loaded.max() < len(labels):
                    X = X_loaded
                    Y = to_categorical(Y_indices_loaded, num_classes=len(labels))
                    if output_text:
                        output_text.insert(END, "Loaded existing preprocessed data (X.npy, Y.npy).\n")
                else:
                    raise ValueError("Loaded data dimensions or label indices seem incorrect.")
            except Exception as e:
                if output_text:
                    output_text.insert(END, f"Error loading existing data or data mismatch ({e}). Reprocessing...\n")
                X, Y = None, None # Force reprocessing

        # If data wasn't loaded, process images
        if X is None or Y is None:
            if output_text:
                output_text.insert(END, "\nProcessing images from scratch...\n")
            processed_count = 0
            skipped_count = 0
            for label_index, label_name in enumerate(labels):
                category_path = os.path.join(filename, label_name)
                if os.path.isdir(category_path):
                    if output_text:
                        output_text.insert(END, f"Processing category: {label_name}...\n")
                        main.update_idletasks()
                    num_in_cat = 0
                    for file in os.listdir(category_path):
                        # Check for valid image extensions
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                            img_path = os.path.join(category_path, file)
                            try:
                                img = cv2.imread(img_path)
                                # Handle cases where image reading fails
                                if img is None:
                                    print(f"Warning: Could not read image {img_path}. Skipping.")
                                    skipped_count += 1
                                    continue

                                # Resize image
                                img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                                # Ensure 3 channels (convert grayscale or RGBA to BGR)
                                if len(img_resized.shape) == 2: # Grayscale
                                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
                                elif img_resized.shape[2] == 4: # BGRA/RGBA
                                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2BGR)

                                # Check if still 3 channels after conversion
                                if img_resized.shape[2] != 3:
                                     print(f"Warning: Image {img_path} has unexpected channel count {img_resized.shape[2]} after conversion. Skipping.")
                                     skipped_count += 1
                                     continue

                                X_data.append(img_resized)
                                Y_data.append(label_index)
                                processed_count += 1
                                num_in_cat +=1
                            except Exception as e:
                                print(f"Error processing image {img_path}: {e}. Skipping.")
                                skipped_count += 1
                                continue
                    if output_text:
                         output_text.insert(END, f"  Processed {num_in_cat} images for {label_name}.\n")

            if not X_data:
                messagebox.showerror("Error", "No valid images were processed. Check dataset content and image formats.")
                return

            if output_text:
                 output_text.insert(END, f"\nTotal images successfully processed: {processed_count}\n")
                 if skipped_count > 0:
                     output_text.insert(END, f"Total images skipped due to errors: {skipped_count}\n")

            # Convert to numpy arrays and normalize
            X = np.array(X_data, dtype='float32') / 255.0
            Y_indices = np.array(Y_data, dtype='int32')
            Y = to_categorical(Y_indices, num_classes=len(labels)) # One-hot encode

            # Save the processed data
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            np.save(x_path, X)
            np.save(y_path, Y_indices) # Save indices, not one-hot Y
            if output_text: output_text.insert(END, f"Saved processed data to {x_path} and {y_path}\n")

        # Split data into Train and Test sets
        if X.shape[0] < 2: # Need at least 2 samples for train/test split
            messagebox.showerror("Error", "Not enough samples processed to split into training and testing sets.")
            X, Y = None, None # Reset X, Y as split failed
            return

        # Ensure test size is reasonable (e.g., at least 1 sample in test set)
        test_size = 0.2
        if int(X.shape[0] * test_size) < 1:
            test_size = 1 / X.shape[0] # Use at least one sample for testing if dataset is tiny
            if output_text: output_text.insert(END, f"Warning: Small dataset size. Adjusting test split to use 1 sample.\n")


        # Stratify ensures proportion of labels is similar in train/test sets
        y_indices_for_split = np.argmax(Y, axis=1) # Get class indices from one-hot Y
        try:
             X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=42, stratify=y_indices_for_split
             )
        except ValueError as e:
             # This can happen if a class has only 1 sample, making stratification impossible
             messagebox.showwarning("Split Warning", f"Could not stratify train/test split (likely due to very few samples in some classes): {e}. Splitting without stratification.")
             X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=test_size, random_state=42 # No stratify
             )


        if output_text:
            output_text.insert(END, "\n--- Dataset Preprocessing Completed ---\n")
            output_text.insert(END, f"Total images: {X.shape[0]}\n")
            output_text.insert(END, f"Image shape: {X.shape[1:]}\n")
            output_text.insert(END, f"Number of categories: {len(labels)}\n")
            output_text.insert(END, f"Training set size: {X_train.shape[0]}\n")
            output_text.insert(END, f"Testing set size: {X_test.shape[0]}\n")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during processing: {e}")
        if output_text:
            output_text.insert(END, f"\nCritical Error during processing: {e}\n")
        # Reset global vars on critical error
        X, Y, X_train, y_train, X_test, y_test = None, None, None, None, None, None

def train_page():
    global output_text
    add_navigation_buttons(train_page)

    # Title
    title_frame = Frame(main, bg="white", bd=0)
    title_frame.place(relx=0.5, rely=0.15, anchor="center")
    title = Label(title_frame, text="Train Models", font=title_font, bg="white", fg="#333333")
    title.pack(padx=10, pady=5)

    # Train Button
    train_frame = Frame(main, bg="#4CAF50", bd=0)
    train_frame.place(relx=0.5, rely=0.3, anchor="center")
    train_button = Button(train_frame, text="Train Models", font=button_font, bg="#4CAF50", fg="white", bd=0, padx=20, pady=10, relief="flat", command=trainModels)
    train_button.pack()

    # Text Area
    text_frame = Frame(main, bd=1, relief="solid")
    text_frame.place(relx=0.5, rely=0.65, relwidth=0.8, relheight=0.5, anchor="center")
    output_text = Text(text_frame, font=text_font, bg="white", fg="#333333", wrap=WORD, bd=0)
    output_text.pack(padx=1, pady=1, fill="both", expand=True)

    # Display status
    if classifier or resnet_model: # Check if models are already in memory
        output_text.insert(END, "Models seem to be already trained or loaded in this session.\n")
        output_text.insert(END, "Clicking 'Train Models' again will re-train and potentially overwrite saved files.\n")
    elif X_train is None or y_train is None:
        output_text.insert(END, "Preprocessed training data is not available.\n")
        output_text.insert(END, "Please go back to the 'Process' page and click 'Preprocess Dataset'.\n")
    else:
        output_text.insert(END, "Ready to train models.\n\n")
        output_text.insert(END, f"Training data shape: {X_train.shape}\n")
        output_text.insert(END, f"Testing data shape: {X_test.shape}\n")
        output_text.insert(END, f"Number of categories: {len(labels)}\n\n")
        output_text.insert(END, "This will train two models:\n")
        output_text.insert(END, "1. A Custom Convolutional Neural Network (CNN).\n")
        output_text.insert(END, "2. A ResNet50 model using Transfer Learning.\n\n")
        output_text.insert(END, "Training progress will be printed here, and evaluation results/graphs will be saved to the 'model' folder.\n")
        if not PLOT_AVAILABLE:
             output_text.insert(END, "\nWarning: Matplotlib/Seaborn not found. Graphs (confusion matrix, ROC) will not be generated.\n")

def trainModels():
    global classifier, resnet_model, labels, X_train, y_train, X_test, y_test, output_text

    if X_train is None or y_train is None or X_test is None or y_test is None:
        messagebox.showerror("Error", "Training/Testing data not available. Please process the dataset first.")
        return
    if not labels:
        messagebox.showerror("Error", "Labels are missing. Please re-upload or process the dataset.")
        return

    if output_text:
        output_text.delete(1.0, END)
        output_text.insert(END, "Starting model training process...\n")
        output_text.insert(END, f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}\n")
        output_text.insert(END, f"Number of Classes: {len(labels)}\n")
        main.update_idletasks()

    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- Train Custom CNN ---
    cnn_history_data = None
    try:
        output_text.insert(END, "\n=== Training Custom CNN ===\n")
        main.update_idletasks()

        cnn_input_shape = X_train.shape[1:] # (64, 64, 3)
        num_classes = len(labels)

        # Build CNN Model
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=cnn_input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5), # Regularization
            Dense(num_classes, activation='softmax') # Output layer for classification
        ])

        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        output_text.insert(END, "Custom CNN Model Summary:\n")
        cnn_model.summary(print_fn=lambda x: output_text.insert(END, x + '\n'))
        main.update_idletasks()


        # Checkpoint to save the best model based on validation accuracy
        cnn_model_path = os.path.join(model_dir, "logo_cnn_best.keras")
        cnn_checkpoint = ModelCheckpoint(
            cnn_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        output_text.insert(END, "\nStarting CNN Training...\n")
        main.update_idletasks()

        # Train the model
        history = cnn_model.fit(
            X_train, y_train,
            batch_size=32, # Adjust batch size based on memory
            epochs=10, # Increase epochs for potentially better results
            validation_data=(X_test, y_test),
            callbacks=[cnn_checkpoint],
            verbose=1 # Show progress bar
        )
        cnn_history_data = history.history # Store history for plotting/saving

        output_text.insert(END, "\nCNN Training Finished.\n")
        output_text.insert(END, f"Best CNN model saved to {cnn_model_path}\n")

        # Load the best saved model for evaluation
        if os.path.exists(cnn_model_path):
             output_text.insert(END, "Loading best saved CNN model for evaluation...\n")
             classifier = load_model(cnn_model_path)
        else:
             output_text.insert(END, "Warning: Best model file not found. Using the last epoch model.\n")
             classifier = cnn_model # Fallback to the model in memory


        # Evaluate CNN
        output_text.insert(END, "Evaluating CNN model on test data...\n")
        loss, acc = classifier.evaluate(X_test, y_test, verbose=0)
        acc = acc * 100 # Convert to percentage

        # Get predictions for detailed metrics
        cnn_probs = classifier.predict(X_test)
        cnn_preds = np.argmax(cnn_probs, axis=1)
        true_labels = np.argmax(y_test, axis=1)

        precision = precision_score(true_labels, cnn_preds, average='macro', zero_division=0) * 100
        recall = recall_score(true_labels, cnn_preds, average='macro', zero_division=0) * 100
        f1 = f1_score(true_labels, cnn_preds, average='macro', zero_division=0) * 100

        output_text.insert(END, "\n--- Custom CNN Test Results ---\n")
        output_text.insert(END, f"Accuracy:  {acc:.2f}%\n")
        output_text.insert(END, f"Precision: {precision:.2f}%\n")
        output_text.insert(END, f"Recall:    {recall:.2f}%\n")
        output_text.insert(END, f"F1 Score:  {f1:.2f}%\n")

        # --- CNN Classification Report ---
        try:
            report = classification_report(true_labels, cnn_preds, target_names=labels, zero_division=0)
            output_text.insert(END, "\n--- CNN Classification Report ---\n")
            output_text.insert(END, report + "\n")
        except Exception as e:
            output_text.insert(END, f"\nError generating CNN classification report: {e}\n")

        # Save CNN history
        try:
            history_path = os.path.join(model_dir, 'history.pkl')
            with open(history_path, 'wb') as f:
                pickle.dump(cnn_history_data, f)
            output_text.insert(END, f"CNN training history saved to {history_path}\n")
        except Exception as e:
             output_text.insert(END, f"Error saving CNN history: {e}\n")


        # --- Plotting CNN Metrics (if libraries available) ---
        if PLOT_AVAILABLE:
            output_text.insert(END, "Generating CNN evaluation plots...\n")
            # Confusion Matrix
            try:
                cm = confusion_matrix(true_labels, cnn_preds)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('CNN Confusion Matrix')
                plt.tight_layout()
                cnn_cm_path = os.path.join(model_dir, 'cnn_confusion_matrix.png')
                plt.savefig(cnn_cm_path)
                plt.close()
                output_text.insert(END, f"CNN Confusion Matrix saved to {cnn_cm_path}\n")
            except Exception as e:
                output_text.insert(END, f"Error generating CNN Confusion Matrix: {e}\n")

            # ROC Curve (One-vs-Rest)
            try:
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                plt.figure(figsize=(10, 8))
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']) # Add more colors if needed

                for i, color in zip(range(num_classes), colors):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], cnn_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                             label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:0.2f})')

                plt.plot([0, 1], [0, 1], 'k--', lw=2) # Diagonal line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('CNN Multi-class ROC Curve (One-vs-Rest)')
                plt.legend(loc="lower right")
                plt.tight_layout()
                cnn_roc_path = os.path.join(model_dir, 'cnn_roc_curve.png')
                plt.savefig(cnn_roc_path)
                plt.close()
                output_text.insert(END, f"CNN ROC curve saved to {cnn_roc_path}\n")

            except Exception as e:
                output_text.insert(END, f"Error generating CNN ROC Curve: {e}\n")
        else:
            output_text.insert(END, "Skipping CNN plot generation (Matplotlib/Seaborn not installed).\n")


    except Exception as e:
        messagebox.showerror("CNN Training Error", f"Failed to train Custom CNN: {e}")
        output_text.insert(END, f"\n\nFATAL ERROR during Custom CNN Training: {e}\n")
        classifier = None # Ensure classifier is None if training failed


    # --- Train ResNet50 ---
    resnet_history_data = None
    try:
        output_text.insert(END, "\n\n=== Training ResNet50 (Transfer Learning) ===\n")
        main.update_idletasks()

        resnet_input_shape = X_train.shape[1:] # (64, 64, 3)
        num_classes = len(labels)

        # Load ResNet50 base model (pre-trained on ImageNet) - Ensure input shape matches
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=resnet_input_shape)
        base_model.trainable = False # Freeze the convolutional base

        # Build the full ResNet model
        resnet_model_full = Sequential([
            base_model,
            GlobalAveragePooling2D(), # Pool features
            Dense(256, activation='relu'), # Add a dense layer
            Dropout(0.5), # Regularization
            Dense(num_classes, activation='softmax') # Output layer
        ])

        resnet_model_full.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        output_text.insert(END, "ResNet50 Model Summary (Top Layers):\n")
        resnet_model_full.summary(print_fn=lambda x: output_text.insert(END, x + '\n'))
        main.update_idletasks()

        # Checkpoint for ResNet
        resnet_model_path = os.path.join(model_dir, "resnet50_best.keras")
        resnet_checkpoint = ModelCheckpoint(
            resnet_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        output_text.insert(END, "\nStarting ResNet50 Training...\n")
        main.update_idletasks()

        # Train the ResNet model
        resnet_history = resnet_model_full.fit(
            X_train, y_train,
            batch_size=32,
            epochs=10, # Can be fewer epochs for transfer learning
            validation_data=(X_test, y_test),
            callbacks=[resnet_checkpoint],
            verbose=1
        )
        resnet_history_data = resnet_history.history

        output_text.insert(END, "\nResNet50 Training Finished.\n")
        output_text.insert(END, f"Best ResNet50 model saved to {resnet_model_path}\n")

        # Load the best saved ResNet model
        if os.path.exists(resnet_model_path):
            output_text.insert(END, "Loading best saved ResNet50 model for evaluation...\n")
            resnet_model = load_model(resnet_model_path) # Assign to global resnet_model
        else:
            output_text.insert(END, "Warning: Best ResNet model file not found. Using the last epoch model.\n")
            resnet_model = resnet_model_full # Assign the one in memory

        # Evaluate ResNet50
        output_text.insert(END, "Evaluating ResNet50 model on test data...\n")
        resnet_loss, resnet_acc = resnet_model.evaluate(X_test, y_test, verbose=0)
        resnet_acc = resnet_acc * 100

        # Get ResNet predictions
        resnet_probs = resnet_model.predict(X_test)
        resnet_preds = np.argmax(resnet_probs, axis=1)
        # true_labels are the same as before

        resnet_precision = precision_score(true_labels, resnet_preds, average='macro', zero_division=0) * 100
        resnet_recall = recall_score(true_labels, resnet_preds, average='macro', zero_division=0) * 100
        resnet_f1 = f1_score(true_labels, resnet_preds, average='macro', zero_division=0) * 100

        output_text.insert(END, "\n--- ResNet50 Test Results ---\n")
        output_text.insert(END, f"Accuracy:  {resnet_acc:.2f}%\n")
        output_text.insert(END, f"Precision: {resnet_precision:.2f}%\n")
        output_text.insert(END, f"Recall:    {resnet_recall:.2f}%\n")
        output_text.insert(END, f"F1 Score:  {resnet_f1:.2f}%\n")

        # --- ResNet Classification Report ---
        try:
            report = classification_report(true_labels, resnet_preds, target_names=labels, zero_division=0)
            output_text.insert(END, "\n--- ResNet50 Classification Report ---\n")
            output_text.insert(END, report + "\n")
        except Exception as e:
            output_text.insert(END, f"\nError generating ResNet classification report: {e}\n")


        # Save ResNet history
        try:
             resnet_history_path = os.path.join(model_dir, 'resnet_history.pkl')
             with open(resnet_history_path, 'wb') as f:
                 pickle.dump(resnet_history_data, f)
             output_text.insert(END, f"ResNet50 training history saved to {resnet_history_path}\n")
        except Exception as e:
             output_text.insert(END, f"Error saving ResNet history: {e}\n")


        # --- Plotting ResNet Metrics (if libraries available) ---
        if PLOT_AVAILABLE:
            output_text.insert(END, "Generating ResNet50 evaluation plots...\n")
            # Confusion Matrix
            try:
                cm = confusion_matrix(true_labels, resnet_preds)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels) # Different color map
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('ResNet50 Confusion Matrix')
                plt.tight_layout()
                resnet_cm_path = os.path.join(model_dir, 'resnet_confusion_matrix.png')
                plt.savefig(resnet_cm_path)
                plt.close()
                output_text.insert(END, f"ResNet50 Confusion Matrix saved to {resnet_cm_path}\n")
            except Exception as e:
                output_text.insert(END, f"Error generating ResNet50 Confusion Matrix: {e}\n")

            # ROC Curve
            try:
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                plt.figure(figsize=(10, 8))
                colors = cycle(['blue', 'red', 'green', 'darkorange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']) # Slightly different colors

                for i, color in zip(range(num_classes), colors):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], resnet_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                             label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:0.2f})')

                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ResNet50 Multi-class ROC Curve (One-vs-Rest)')
                plt.legend(loc="lower right")
                plt.tight_layout()
                resnet_roc_path = os.path.join(model_dir, 'resnet_roc_curve.png')
                plt.savefig(resnet_roc_path)
                plt.close()
                output_text.insert(END, f"ResNet50 ROC curve saved to {resnet_roc_path}\n")
            except Exception as e:
                 output_text.insert(END, f"Error generating ResNet ROC Curve: {e}\n")
        else:
             output_text.insert(END, "Skipping ResNet50 plot generation (Matplotlib/Seaborn not installed).\n")

    except Exception as e:
        messagebox.showerror("ResNet Training Error", f"Failed to train ResNet50: {e}")
        output_text.insert(END, f"\n\nFATAL ERROR during ResNet50 Training: {e}\n")
        resnet_model = None # Ensure resnet_model is None if training failed

    # --- Model Comparison Plot (Overall Metrics) ---
    if PLOT_AVAILABLE and classifier and resnet_model: # Check if both models were trained successfully
        output_text.insert(END, "\nGenerating Model Comparison plot...\n")
        try:
            plt.figure(figsize=(12, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            # Use the calculated metrics directly
            cnn_scores = [acc , precision , recall , f1 ]
            resnet_scores = [resnet_acc, resnet_precision, resnet_recall, resnet_f1]

            x = np.arange(len(metrics))
            width = 0.35 # Bar width

            rects1 = plt.bar(x - width/2, cnn_scores, width, label='Custom CNN', color='skyblue')
            rects2 = plt.bar(x + width/2, resnet_scores, width, label='ResNet50', color='lightcoral')

            plt.ylabel('Scores (%)')
            plt.title('Model Performance Comparison (Macro Averages)')
            plt.xticks(x, metrics)
            plt.ylim(0, 105) # Set Y axis limit slightly above 100
            plt.legend()

            # Add labels on top of bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    plt.annotate(f'{height:.1f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)

            plt.tight_layout()
            comparison_path = os.path.join(model_dir, 'model_comparison.png')
            plt.savefig(comparison_path)
            plt.close()
            output_text.insert(END, f"Model comparison chart saved to {comparison_path}\n")
        except Exception as e:
            output_text.insert(END, f"Error generating comparison chart: {e}\n")
    elif PLOT_AVAILABLE:
         output_text.insert(END, "\nSkipping model comparison plot as one or both models failed to train.\n")


    output_text.insert(END, "\n\n--- Training and Evaluation Complete ---\n")
    messagebox.showinfo("Training Complete", "Model training and evaluation finished. Results and plots (if generated) are in the 'model' folder.")

def result_page():
    global output_text, classifier, resnet_model, labels
    add_navigation_buttons(result_page)

    # Title
    title_frame = Frame(main, bg="white", bd=0)
    title_frame.place(relx=0.5, rely=0.15, anchor="center")
    title = Label(title_frame, text="Classify Logo Image", font=title_font, bg="white", fg="#333333")
    title.pack(padx=10, pady=5)

    # Classify Button
    classify_frame = Frame(main, bg="#4CAF50", bd=0)
    classify_frame.place(relx=0.5, rely=0.3, anchor="center")
    classify_button = Button(classify_frame, text="Upload & Classify Logo", font=button_font, bg="#4CAF50", fg="white", bd=0, padx=20, pady=10, relief="flat", command=classifyLogo)
    classify_button.pack()

    # Text Area
    text_frame = Frame(main, bd=1, relief="solid")
    text_frame.place(relx=0.5, rely=0.65, relwidth=0.8, relheight=0.5, anchor="center")
    output_text = Text(text_frame, font=text_font, bg="white", fg="#333333", wrap=WORD, bd=0)
    output_text.pack(padx=1, pady=1, fill="both", expand=True)

    # --- Load Models and Labels if not already in memory ---
    model_dir = "model"
    cnn_model_path = os.path.join(model_dir, "logo_cnn_best.keras")
    resnet_model_path = os.path.join(model_dir, "resnet50_best.keras")
    labels_path = os.path.join(model_dir, "labels.pkl")

    models_loaded = True

    # Load Labels
    if not labels:
        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'rb') as f:
                    labels = pickle.load(f)
                output_text.insert(END, "Loaded labels from file.\n")
            except Exception as e:
                output_text.insert(END, f"Error loading labels: {e}\n")
                models_loaded = False
        else:
             output_text.insert(END, "Error: Labels file not found. Cannot classify.\n")
             models_loaded = False


    # Load CNN Model
    if not classifier: # Only load if not already in memory
        if os.path.exists(cnn_model_path):
            try:
                classifier = load_model(cnn_model_path)
                output_text.insert(END, "Loaded trained Custom CNN model from file.\n")
            except Exception as e:
                output_text.insert(END, f"Error loading Custom CNN model: {e}\nPlease ensure the model file is not corrupted and dependencies match.\n")
                models_loaded = False
        else:
             output_text.insert(END, "Error: Trained Custom CNN model file not found.\n")
             models_loaded = False

    # Load ResNet Model
    if not resnet_model: # Only load if not already in memory
        if os.path.exists(resnet_model_path):
            try:
                resnet_model = load_model(resnet_model_path)
                output_text.insert(END, "Loaded trained ResNet50 model from file.\n")
            except Exception as e:
                output_text.insert(END, f"Error loading ResNet50 model: {e}\nPlease ensure the model file is not corrupted and dependencies match.\n")
                models_loaded = False
        else:
             output_text.insert(END, "Error: Trained ResNet50 model file not found.\n")
             models_loaded = False

    # Final Status Message
    if models_loaded and labels and classifier and resnet_model:
        output_text.insert(END, "\nBoth models and labels are ready.\n")
        output_text.insert(END, f"Available categories: {', '.join(labels)}\n")
        output_text.insert(END, "\nClick 'Upload & Classify Logo' to test an image.\n")
    else:
         output_text.insert(END, "\nError: Could not load necessary models or labels. Please ensure dataset was processed and models were trained successfully.\n")


def classifyLogo():
    global classifier, resnet_model, labels, output_text

    if not classifier or not resnet_model:
        messagebox.showerror("Error", "One or both models are not loaded. Please ensure training was successful or try navigating back and forth to reload.")
        return
    if not labels:
        messagebox.showerror("Error", "Labels are not loaded. Cannot determine class names.")
        return

    img_filename = filedialog.askopenfilename(
        title="Select Logo Image for Classification",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
    )
    if not img_filename: # User cancelled selection
        return

    if output_text:
        output_text.delete(1.0, END)
        output_text.insert(END, f"Classifying image: {os.path.basename(img_filename)}\n\n")
        main.update_idletasks()

    try:
        # --- Image Preprocessing (same as during training) ---
        img = cv2.imread(img_filename)
        if img is None:
            messagebox.showerror("Error", f"Could not read the selected image file:\n{img_filename}")
            return

        # Resize
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        # Ensure 3 channels
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2BGR)

        # Check channels again after potential conversion
        if img_resized.shape[2] != 3:
             messagebox.showerror("Error", f"Image has unexpected channel count ({img_resized.shape[2]}) after processing.")
             return

        # Normalize and add batch dimension
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0) # Shape becomes (1, 64, 64, 3)

        # --- CNN Prediction ---
        cnn_preds_proba = classifier.predict(img_batch)[0] # Get probability distribution for the single image
        cnn_index = np.argmax(cnn_preds_proba)
        cnn_confidence = cnn_preds_proba[cnn_index] * 100
        cnn_label = labels[cnn_index] if cnn_index < len(labels) else "Unknown Index"

        # --- ResNet Prediction ---
        resnet_preds_proba = resnet_model.predict(img_batch)[0]
        resnet_index = np.argmax(resnet_preds_proba)
        resnet_confidence = resnet_preds_proba[resnet_index] * 100
        resnet_label = labels[resnet_index] if resnet_index < len(labels) else "Unknown Index"

        # --- Display Results in Text Area ---
        output_text.insert(END, "--- Classification Results ---\n")
        output_text.insert(END, f"Custom CNN Prediction:\n  Label: {cnn_label}\n  Confidence: {cnn_confidence:.2f}%\n\n")
        output_text.insert(END, f"ResNet50 Prediction:\n  Label: {resnet_label}\n  Confidence: {resnet_confidence:.2f}%\n")

        # --- Display image with predictions using OpenCV window ---
        # Load original image again for display to avoid showing the tiny resized one
        display_img = cv2.imread(img_filename)
        # Resize for display purposes if it's too large
        max_display_height = 500
        h, w = display_img.shape[:2]
        if h > max_display_height:
            ratio = max_display_height / h
            display_img = cv2.resize(display_img, (int(w * ratio), max_display_height), interpolation=cv2.INTER_AREA)

        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        cnn_text = f"CNN: {cnn_label} ({cnn_confidence:.1f}%)"
        resnet_text = f"ResNet50: {resnet_label} ({resnet_confidence:.1f}%)"

        # Add text with background for better visibility
        (w_cnn, h_cnn), _ = cv2.getTextSize(cnn_text, font, font_scale, 2)
        cv2.rectangle(display_img, (5, 5), (15 + w_cnn, 25 + h_cnn), (0,0,0), -1) # Black background rect
        cv2.putText(display_img, cnn_text, (10, 20 + h_cnn), font, font_scale, (0, 255, 0), 2) # Green text

        (w_res, h_res), _ = cv2.getTextSize(resnet_text, font, font_scale, 2)
        cv2.rectangle(display_img, (5, 35 + h_cnn), (15 + w_res, 55 + h_cnn + h_res), (0,0,0), -1) # Black background rect
        cv2.putText(display_img, resnet_text, (10, 50 + h_cnn + h_res), font, font_scale, (0, 0, 255), 2) # Red text


        cv2.imshow(f"Classification Result: {os.path.basename(img_filename)}", display_img)
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows() # Close the OpenCV window

    except Exception as e:
        messagebox.showerror("Classification Error", f"An error occurred during classification: {e}")
        if output_text:
            output_text.insert(END, f"\n\nError during classification: {e}\n")
        cv2.destroyAllWindows() # Ensure any OpenCV windows are closed on error

# --- Start Application ---
if __name__ == "__main__":
    # Ensure model directory exists on startup
    if not os.path.exists("model"):
        os.makedirs("model")
    # Start with the home page
    home_page()
    # Start the Tkinter event loop
    main.mainloop()