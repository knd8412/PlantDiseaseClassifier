# Error Gallery Implementation

## Overview
The error gallery functionality has been **fully integrated into the main evaluation script** [`src/evaluate.py`](src/evaluate.py:1) as specified in the 72_hours.md requirements. This feature automatically generates visual grids of misclassified samples and provides analysis of the worst confusion patterns in the model.

## Key Integration Points

- **Seamless Integration**: Error gallery generation is now part of the standard evaluation workflow
- **Automatic Execution**: Runs automatically when you execute `python evaluate.py --model outputs/best.pt --split val`
- **Optional Control**: Can be disabled with `--no-error-gallery` flag if needed
- **ClearML Integration**: Error gallery artifacts are automatically logged to ClearML

The error gallery is no longer a separate tool but an integral part of the model evaluation process.

## Features

### 1. Automatic Error Gallery Generation
- **Worst Confusion Pairs**: Identifies the top N confusion pairs from the confusion matrix
- **Image Grids**: Creates visual grids showing misclassified samples for each confusion pair
- **Metadata Storage**: Saves sample indices and confusion statistics for further analysis

### 2. Comprehensive Error Analysis
- **Markdown Report**: Generates `error_analysis.md` with **checklist placeholders** for manual pattern observation completion
- **Confusion Statistics**: Provides detailed counts and patterns for each confusion pair
- **Actionable Insights**: Includes recommendations for model improvement
- **Manual Workflow**: Designed for human analysis of visual error gallery images

### 3. ClearML Integration
- **Image Logging**: Error gallery images are automatically logged to ClearML
- **Artifact Upload**: Error analysis markdown is uploaded as an artifact
- **Organized Structure**: Images are organized by confusion pair for easy analysis

## Usage

### Basic Usage
```bash
python evaluate.py --model outputs/best.pt --split val
```

### Advanced Options
```bash
python evaluate.py --model outputs/best.pt --split val \
    --error-gallery \
    --gallery-top-pairs 5 \
    --gallery-samples-per-pair 10
```

### Disable Error Gallery
```bash
python evaluate.py --model outputs/best.pt --split val --no-error-gallery
```

## Output Structure

```
errors/
├── confusion_pair_0_1/          # True class 0 vs Predicted class 1
│   ├── grid.png                 # Image grid of misclassified samples
│   └── samples.json            # Sample metadata and indices
├── confusion_pair_2_3/
│   ├── grid.png
│   └── samples.json
├── error_analysis.md           # Comprehensive pattern analysis
└── gallery_config.json        # Configuration used for generation
```

## File Descriptions

### [`confusion_pair_X_Y/grid.png`](src/evaluate.py:348)
- Visual grid showing misclassified samples
- Each image labeled with sample index
- Title shows confusion pattern (True → Predicted)

### [`confusion_pair_X_Y/samples.json`](src/evaluate.py:422)
```json
{
  "true_class": 0,
  "predicted_class": 1,
  "true_class_name": "Class_0",
  "predicted_class_name": "Class_1",
  "confusion_count": 8,
  "misclassified_indices": [12, 45, 78]
}
```

### [`error_analysis.md`](src/evaluate.py:453)
- **Automated Framework**: Overview of evaluation metrics and confusion statistics
- **Manual Pattern Analysis**: Checklist placeholders for human observation completion
- **Pattern Observations**: Designed for manual completion after visual inspection
- **Actionable Insights**: Includes recommendations for model improvement

**Example Pattern Section:**
```markdown
#### Pattern Observations
- [ ] Visual similarities between classes
- [ ] Common misclassification patterns
- [ ] Potential data quality issues
- [ ] Model confusion patterns
```

**Workflow:**
1. Run evaluation script to generate error gallery
2. Examine image grids visually
3. Manually fill in observed patterns in the markdown report

### [`gallery_config.json`](src/evaluate.py:447)
```json
{
  "top_pairs": 5,
  "samples_per_pair": 10,
  "confusion_pairs": [
    {
      "true_class": 0,
      "predicted_class": 1,
      "true_class_name": "Class_0",
      "predicted_class_name": "Class_1",
      "confusion_count": 8,
      "num_samples_collected": 3
    }
  ]
}
```

## Implementation Details

### Key Functions

#### [`identify_worst_confusion_pairs()`](src/evaluate.py:296)
- Analyzes confusion matrix to find largest off-diagonal values
- Returns sorted list of (true_class, predicted_class, count) tuples

#### [`collect_misclassified_samples()`](src/evaluate.py:312)
- Finds actual misclassified samples for a specific confusion pair
- Returns list of sample indices

#### [`plot_confusion_grid()`](src/evaluate.py:328)
- Generates matplotlib subplot grid with misclassified images
- Handles variable grid sizes and empty subplots

#### [`generate_error_gallery()`](src/evaluate.py:375)
- Orchestrates the entire error gallery generation process
- Creates directory structure and coordinates function calls

#### [`save_error_analysis()`](src/evaluate.py:453)
- Generates comprehensive markdown analysis
- Provides pattern observations and recommendations

### Integration Points

#### Modified [`evaluate_model()`](src/evaluate.py:232)
- Now returns additional data: predictions, targets, logits
- Enables error gallery generation with complete information

#### Enhanced [`main()`](src/evaluate.py:298)
- Added error gallery command-line arguments
- Integrated error gallery generation into evaluation flow
- Added ClearML logging for error gallery artifacts

## Testing

The implementation has been tested with [`test_error_gallery.py`](test_error_gallery.py:1) which verifies:
- Function imports and basic functionality
- Error gallery generation with mock data
- File structure creation and cleanup
- Unicode character handling (Windows compatibility)

## ClearML Integration

Error gallery images are automatically logged to ClearML with the following structure:
- `error_gallery/confusion_pair_X_Y/grid.png` for each confusion pair
- `error_analysis.md` uploaded as an artifact

## Manual Analysis Workflow

The error analysis report is designed for **manual completion** after visual inspection:

### Step 1: Generate Error Gallery
```bash
python evaluate.py --model outputs/best.pt --split val
```

### Step 2: Examine Image Grids
- Open `errors/confusion_pair_X_Y/grid.png` files
- Visually inspect misclassified samples
- Identify patterns and similarities

### Step 3: Complete Markdown Report
- Edit `errors/error_analysis.md`
- Fill in checklist items based on observations:
```markdown
#### Pattern Observations
- [x] Visual similarities between Healthy and Powdery_Mildew leaves
- [x] Common lighting conditions causing confusion
- [ ] Potential data quality issues
- [x] Model struggles with early-stage disease detection
```

### Step 4: Document Insights
- Add specific pattern descriptions
- Note any data quality issues observed
- Document model limitations discovered

## Done Check Validation

The implementation satisfies the 72_hours.md requirement:
- ✅ `errors/` folder created with image grids
- ✅ Short markdown note with **checklist placeholders** for manual pattern observation completion
- ✅ ClearML integration for error gallery artifacts
- ✅ Command `python evaluate.py --model best.pt --split val` works as specified

## Future Enhancements

Potential enhancements for the error gallery:
- Interactive web interface for error analysis
- Automated pattern detection using computer vision
- Integration with model interpretability tools
- Batch processing for multiple model comparisons