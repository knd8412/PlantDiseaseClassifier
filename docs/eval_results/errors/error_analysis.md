# Error Analysis Report

## Overview

- Overall Accuracy: 0.9971
- Top-5 Accuracy: 1.0000
- Number of Classes: 38

## Worst Confusion Pairs

### 1. Grape___Black_rot -> Grape___Esca_(Black_Measles) (Count: 1)

- **True Class**: Grape___Black_rot
- **Predicted Class**: Grape___Esca_(Black_Measles)
- **Confusion Count**: 1

#### Pattern Observations

- [ ] Visual similarities between classes
- [ ] Common misclassification patterns
- [ ] Potential data quality issues
- [ ] Model confusion patterns

### 2. Potato___Late_blight -> Tomato___Late_blight (Count: 1)

- **True Class**: Potato___Late_blight
- **Predicted Class**: Tomato___Late_blight
- **Confusion Count**: 1

#### Pattern Observations

- [ ] Visual similarities between classes
- [ ] Common misclassification patterns
- [ ] Potential data quality issues
- [ ] Model confusion patterns

### 3. Tomato___Early_blight -> Tomato___Late_blight (Count: 1)

- **True Class**: Tomato___Early_blight
- **Predicted Class**: Tomato___Late_blight
- **Confusion Count**: 1

#### Pattern Observations

- [ ] Visual similarities between classes
- [ ] Common misclassification patterns
- [ ] Potential data quality issues
- [ ] Model confusion patterns

### 4. Tomato___Late_blight -> Pepper,_bell___healthy (Count: 1)

- **True Class**: Tomato___Late_blight
- **Predicted Class**: Pepper,_bell___healthy
- **Confusion Count**: 1

#### Pattern Observations

- [ ] Visual similarities between classes
- [ ] Common misclassification patterns
- [ ] Potential data quality issues
- [ ] Model confusion patterns

### 5. Tomato___Late_blight -> Tomato___Septoria_leaf_spot (Count: 1)

- **True Class**: Tomato___Late_blight
- **Predicted Class**: Tomato___Septoria_leaf_spot
- **Confusion Count**: 1

#### Pattern Observations

- [ ] Visual similarities between classes
- [ ] Common misclassification patterns
- [ ] Potential data quality issues
- [ ] Model confusion patterns

### 6. Tomato___Target_Spot -> Tomato___Septoria_leaf_spot (Count: 1)

- **True Class**: Tomato___Target_Spot
- **Predicted Class**: Tomato___Septoria_leaf_spot
- **Confusion Count**: 1

#### Pattern Observations

- [ ] Visual similarities between classes
- [ ] Common misclassification patterns
- [ ] Potential data quality issues
- [ ] Model confusion patterns

## Recommendations

- Consider data augmentation for frequently confused classes
- Review class balance and dataset quality
- Evaluate model architecture for class discrimination
- Consider transfer learning or fine-tuning approaches
