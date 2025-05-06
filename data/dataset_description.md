# Dataset Description

## 1. Kaggle Brain MRI Dataset

The Kaggle dataset consists of 7023 human brain MRI images, classified into four distinct categories:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

### Dataset Details:

- **Format**: JPG images
- **Resolution**: 512x512 pixels
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Total Images**: 7023
- **Training Images**: 5712
- **Testing Images**: 1311
- **Data Split**: 80% for training, 20% for testing (which also serves as the validation dataset)

The dataset is used for **classification** tasks to detect and categorize brain tumors based on MRI scans.

## 2. BraTS 2020 Dataset

The **BraTS 2020** (Brain Tumor Segmentation) challenge dataset focuses on the segmentation of brain tumors, particularly **gliomas**, from multimodal MRI scans.

### Dataset Details:

- **Format**: NIfTI (.nii) format
- **Modalities**:
  - **T1-weighted (T1)**: Shows anatomical details and contrast between various tissues.
  - **Post-contrast T1-weighted (T1Ce)**: Enhanced images after gadolinium contrast injection to highlight abnormal blood-brain barrier permeability.
  - **T2-weighted (T2)**: Amplifies fluids, useful for detecting edema or swelling.
  - **FLAIR (Fluid-Attenuated Inversion Recovery)**: Suppresses fluid signals, helping to identify lesions near fluid spaces.

- **Segmentation Classes**:
  - **No Tumor**
  - **Necrotic Core**
  - **Enhancing Tumor**
  - **Edema Core**

The dataset includes **segmented images** that provide pixel-based anatomical labels of the tumor sub-regions, making it invaluable for **tumor segmentation** tasks.

### Collection Details:
- Collected from **19 different institutions** using various clinical protocols and MRI scanners.
- Each dataset underwent **manual segmentation** by one to four raters, with final approval by experienced **neuroradiologists**.

The BraTS dataset is used primarily for **segmentation tasks** in brain tumor detection.

## Conclusion

- **Kaggle Dataset**: Primarily used for **classification** of different brain tumor types.
- **BraTS 2020 Dataset**: Used for **segmentation** of glioma tumors, with detailed tumor sub-region labels, including **necrotic core**, **enhancing tumor**, and **edema core**.

Both datasets combined support research in **automated brain tumor detection**, enhancing the ability to improve diagnostic accuracy and assist healthcare professionals in making timely decisions.
