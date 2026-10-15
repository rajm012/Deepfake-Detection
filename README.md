# Deepfake Video Detection: Unsupervised Contrastive Learning

A course project for **CS671: Deep Learning** at IIT Mandi. It replicates and extends *Fully Unsupervised Deepfake Video Detection via Enhanced Contrastive Learning*, whose PDF is included in the repo.

The aim is to tell real videos from manipulated ones **without labelled training data**. Pseudo-labels are generated from facial cues, and a contrastive encoder trained on them separates real faces from fake ones.

---

## Pipeline (`2.Ours-Mod/`)

1. **Stage 1: pseudo-labelling.** Detect faces and 68 landmarks (dlib). Extract **Laws texture-energy** features from the eye and mouth regions, along with eye- and mouth-open cues. Cluster them (PCA → K-Means / GMM) to assign pseudo real/fake labels.
2. **Stage 2: enhanced contrastive learning.** Train an encoder (EfficientNet backbone, with a ViT variant) using supervised-contrastive loss on the pseudo-labels.
3. **Stage 3: inference.** Crop faces with MTCNN, embed the frames, cluster the embeddings and aggregate per-video predictions.

## Repository layout

| Path | Contents |
|---|---|
| `1.Auther-Org/` | The original authors' implementation, used as the baseline |
| `1-Modification/` | First modifications to the contrastive learner |
| `2.Ours-Mod/` | Our three-stage pipeline (Stage1–3, backbones, data transforms, utilities) |
| `3.Our-Result/` | Pseudo-labels (26k / 32k / 800k frames) and predictions on Celeb-DF, FaceForensics++ and UADFV |
| `5.Help/` | Notes and helper material |
| Root `*.py` | Standalone model, loss, binary-classifier and accuracy scripts |
| `*.pptx` | Project presentation |

## Running

```bash
pip install torch torchvision facenet-pytorch dlib opencv-python scikit-learn tqdm
python 2.Ours-Mod/Stage1.py --unlabeled_data_path <videos> --face_detector_path shape_predictor_68_face_landmarks.dat --output_path out/
python 2.Ours-Mod/Stage2.py   # contrastive training on the Stage 1 pseudo-labels
python 2.Ours-Mod/Stage3.py --test_data_path <test_videos> --checkpoint_path <encoder.pth>
```

The datasets (Celeb-DF, FaceForensics++, UADFV) are not included. Download them from their official sources.
