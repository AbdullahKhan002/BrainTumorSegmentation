# Brain Tumor Segmentation

Brain tumor segmentation involves identifying and delineating tumor regions in brain images. This project uses low-quality images for segmentation.

## Installation Instructions

1. Clone this repository:
```bash
git clone https://github.com/AbdullahKhan002/BrainTumorSegmentation.git
```
2. Navigate to the project directory:
```bash
cd BrainTumorSegmentation_Project
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
4.Ensure you have TensorFlow, Albumentations, and OpenCV installed.

## Usage Guide
1. Place your dataset in the appropriate folder structure:
```bash
/data/
├── images/  
└── masks/
```
2.Run the training script:
```bash
python .\src\train.py
```
3.The model and logs will be saved in the ```results``` directory.

**For Testing**:
```bash
python .\src\test.py
```
## Results and Visuals
**Training Results**: The model achieves segmentation accuracy on the test set, visualized with overlayed masks.
