# Skin Cancer Classification using Deep Learning

Deep learning project for classifying skin lesion images into 7 categories using CNN and Transfer Learning with PyTorch.

## Project Overview

This project demonstrates computer vision and deep learning skills for medical image analysis. It uses transfer learning with ResNet18 to classify dermatoscopic images of skin lesions.

**Key Technologies:**
- Python 3.10+
- PyTorch (Deep Learning)
- Torchvision (Pre-trained models)
- ResNet18 (Transfer Learning)
- CUDA/GPU acceleration

## Dataset

**Source:** HAM10000 - Skin Cancer MNIST (Kaggle)  
**Total Images:** 10,015  
**Training:** 8,012 images (80%)  
**Testing:** 2,003 images (20%)  
**Classes:** 7 types of skin lesions  
**Imbalance Ratio:** 58.3x

### Class Distribution

| Code | Description | Count |
|------|-------------|-------|
| nv | Melanocytic nevi | 6,705 |
| mel | Melanoma | 1,113 |
| bkl | Benign keratosis | 1,099 |
| bcc | Basal cell carcinoma | 514 |
| akiec | Actinic keratoses | 327 |
| vasc | Vascular lesions | 142 |
| df | Dermatofibroma | 115 |

## Model Architecture

**Base Model:** ResNet18 (pre-trained on ImageNet)

**Transfer Learning Approach:**
- Frozen early convolutional layers (feature extraction)
- Replaced final fully-connected layer for 7 classes
- Only trained final layer parameters

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss with class weights
- Batch size: 32
- Epochs: 5
- GPU: Tesla T4

## Results

**Overall Performance:**
- **Validation Accuracy:** 65.60%
- **Training Time:** ~8 minutes (5 epochs)

**Per-Class Performance:**
- **Best:** vasc (Vascular lesions) - Highest accuracy
- **Worst:** akiec (Actinic keratoses) - Lowest accuracy
- **Average:** Varies by class due to severe imbalance

**Key Metrics:**
- Handled 58.3x class imbalance using weighted loss
- Applied data augmentation (flip, rotation, color jitter)
- Stratified train-test split maintained class ratios

## Custom Analyses

### 1. Per-Class Performance Analysis
**Research Question:** Which lesion types are easiest/hardest to classify?

**Findings:**
- Vascular lesions (vasc) achieve highest accuracy due to distinct color/pattern
- Actinic keratoses (akiec) have lowest accuracy - visually similar to other lesions
- Performance strongly correlated with visual distinctiveness

### 2. Dataset Size Impact
**Research Question:** How does training sample count affect accuracy?

**Findings:**
- Correlation between sample size and accuracy: [X.XX]
- Classes with more training examples generally perform better
- Demonstrates importance of balanced datasets in medical AI

### 3. Misclassification Patterns
**Research Question:** Which classes are commonly confused?

**Findings:**
- Most common confusion: [Class A] → [Class B]
- Visually similar lesions (similar texture/color) are often misclassified
- Clinical context needed for accurate differential diagnosis

## Technical Highlights

- **Transfer Learning:** Leveraged ImageNet pre-training
- **Class Imbalance:** Computed and applied class weights (up to 58x)
- **Data Augmentation:** Random flips, rotations, color jitter
- **GPU Acceleration:** CUDA-enabled training (~8 min vs hours on CPU)
- **Stratified Splitting:** Maintained class distribution in train/test
- **Error Analysis:** Comprehensive confusion matrix and per-class metrics

## Project Structure
```
skin-cancer-classification/
│
├── Skin_Cancer_Classification.ipynb    # Main notebook
└── README.md                            # This file
```

## How to Run

### Google Colab (Recommended)
1. Upload notebook to Colab
2. **Runtime → Change runtime type → GPU (T4)**
3. Install Kaggle API: `!pip install kaggle`
4. Upload `kaggle.json` (from kaggle.com/settings)
5. Run all cells

### Local Environment
```bash
# Install dependencies
pip install torch torchvision pandas matplotlib seaborn scikit-learn pillow

# Download dataset from Kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Launch Jupyter
jupyter notebook Skin_Cancer_Classification.ipynb
```

**Note:** GPU strongly recommended (8 min vs 2+ hours on CPU)

## Medical Significance

### Clinical Relevance
- Early melanoma detection is life-saving (5-year survival: 99% if caught early)
- AI can assist dermatologists in screening large patient populations
- Reduces diagnostic workload and catches cases human eyes might miss

### Limitations
- 65.6% accuracy insufficient for clinical deployment (needs 90%+)
- Dataset lacks patient metadata (age, skin type, lesion location)
- Single image per lesion (clinical exam uses multiple angles/dermoscopy)
- Severe class imbalance affects rare lesion detection

### Ethical Considerations
- AI should augment, not replace, dermatologist expertise
- False negatives (missing melanoma) are more dangerous than false positives
- Model trained primarily on lighter skin tones (dataset bias)

## Challenges Faced

1. **Severe Class Imbalance (58x):** Solved with class-weighted loss function
2. **Limited GPU Time:** Optimized with transfer learning (only train final layer)
3. **Large Dataset:** Used efficient PyTorch DataLoader with num_workers=2

## Future Improvements

- [ ] Use larger models (ResNet50, EfficientNet-B4)
- [ ] Expand to 100K+ images with better balance
- [ ] Implement ensemble methods (combine multiple models)
- [ ] Add Grad-CAM visualization (show what model "sees")
- [ ] Include patient metadata (age, lesion location)
- [ ] Deploy as web app for real-time predictions
- [ ] Test on diverse skin tones (address dataset bias)

## Skills Demonstrated

- Deep Learning (CNNs, Transfer Learning)
- PyTorch framework
- Computer Vision
- Medical image analysis
- Handling class imbalance
- Data augmentation
- GPU acceleration
- Model evaluation and metrics
- Error analysis
- Technical documentation

## Author

**Hudayi Hamza Adatepe**  
Computer Engineering Student, 3rd Year  
Seeking Summer 2026 Internship

**Contact:**
- LinkedIn: www.linkedin.com/in/hüdayi-adatepe-9073121b8
- Email: dayihamza_60@hotmail.com
- GitHub: @Hudayiadatepe(https://github.com/Hudayiadatepe)

## Related Projects

- [COVID-19 Turkey Analysis](https://github.com/yourusername/covid19-turkey-analysis) - Data analysis
- [Turkish Sentiment Analysis](https://github.com/yourusername/turkish-sentiment-analysis) - NLP

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: HAM10000 (Tschandl et al., 2018)
- PyTorch and Torchvision teams
- ResNet architecture (He et al., 2015)
- Google Colab for free GPU access
