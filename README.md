# 🐢 Tag a Turtle - Current Stakeholder Model Pipeline & Architecture

This pipeline is based on the **current model used by the Sea Turtle Conservation Bonaire** and was originally developed by **Fru/itPunch AI**.

The current model integrates multiple components for automatic sea turtle re-identification:
- **YOLOv8** for **object detection** (locating the turtle in the frame)
- **SAM (Segment Anything Model)** for **segmentation** of the turtle's head
- **LightGlue** for **identifying individual turtles** based on the unique **scale patterns on their heads**

---

## 🧩 Why this Pipeline?

While the model runs in **Google Colab** (where the stakeholder interacts with it), the team required a **modular and local version** of the pipeline to:

- **Enable continuous development and testing**
- **Assess areas for improvement**
- **Compare accuracy with other models**
- **Use more powerful local GPUs without Google Colab constraints**

This modular version allows for more flexibility, better organization, and long-term maintainability.

---

## 📊 Model Architecture

Below, you can see the model architecture with the features mentioned in this pipeline (Yolov8, SAM, Lightglue).

![image](https://i.imgur.com/03W3e9l.png)
---

## 🧪 Environment Setup & Requirements

To run the Current Model pipeline locally, follow the steps below:

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/Turtle-AI-Bonaire/original_model_byfp.git
cd original_model_byfp
```

### 2. 🐍 Create a Virtual Environment (Recommended)

We recommend using a virtual environment to avoid conflicts:

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
```

### 3. 📦 Install Dependencies

Make sure you have Python ≥ 3.8, then install the required packages:

```bash
pip install -r requirements.txt
```

### 4. 🗂 Create Dataset Folder

Inside the root of the project, create a folder to store the dataset:

```bash
mkdir -p data/raw
```
So your structure will look like:
```kotlin
project/
└── data/
    └── raw/
```
### 5. 📁 Upload Annotated Dataset

Download and extract the annotated the Turtle dataset by Sea Turtle Conservation Bonaire into the data/raw/ folder. **Important note:** This dataset is provided on Google Drive and given in the moment of transfer for data privacy reasons.

After extraction, the structure should look like this:

```kotlin
data/raw/
├── 2019/
│ ├── 2019-04-05_13-20-33.jpg
│ ├── 2019-06-21_10-45-12.jpg
│ └── ...
├── 2020/
│ ├── 2020-03-11_09-15-02.jpg
│ └── ...
└── 2021/
└── ...
```
### ▶️ Run the Pipeline

Once the data is in place and dependencies installed, simply run:

```bash
python main.py
```
This will:

- Load the dataset
- Preprocess images
- Use the models to identify a new turtle, or a matched turtle.

🧠 Note: If you're using a GPU, TensorFlow will detect and use it automatically if drivers and CUDA/cuDNN are correctly installed.


