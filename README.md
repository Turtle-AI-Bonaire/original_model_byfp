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

*(Diagram of the original model will be inserted here)*

---

## 🧪 Environment Setup & Requirements
