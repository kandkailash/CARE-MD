# ==========================================================
# IMPORTS
# ==========================================================

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ATTUNET_MODEL,
    HYBRID_MODEL,
    MEMORY_BANK,
    REFERENCE_IMAGE_DIR,
    REPORT_DIR,
    ROI_DIR,
    MASK_DIR,
    CLASS_NAMES,
    CONCEPT_NAMES
)

# ==========================================
# ATTENTION BLOCK
# ==========================================
class AttentionBlock(nn.Module):
    def __init__(self, in_g, in_x, out_c):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_g, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_x, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_c, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# ==========================================
# CONV BLOCK
# ==========================================
def conv_block(in_c, out_c):

        return nn.Sequential(

            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_c,
                out_c,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


# ==========================================
# ATTENTION U-NET
# ==========================================
class AttUNet(nn.Module):

    def __init__(
        self,
        in_c=3,
        out_c=1,
        base_c=64
    ):
        super().__init__()

        # Encoder
        self.c1 = conv_block(in_c, base_c)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = conv_block(base_c, base_c*2)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = conv_block(base_c*2, base_c*4)
        self.p3 = nn.MaxPool2d(2)

        self.c4 = conv_block(base_c*4, base_c*8)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.c5 = conv_block(base_c*8, base_c*16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            base_c*16,
            base_c*8,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionBlock(
            base_c*8,
            base_c*8,
            base_c*4
        )

        self.c6 = conv_block(
            base_c*16,
            base_c*8
        )

        self.up3 = nn.ConvTranspose2d(
            base_c*8,
            base_c*4,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionBlock(
            base_c*4,
            base_c*4,
            base_c*2
        )

        self.c7 = conv_block(
            base_c*8,
            base_c*4
        )

        self.up2 = nn.ConvTranspose2d(
            base_c*4,
            base_c*2,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionBlock(
            base_c*2,
            base_c*2,
            base_c
        )

        self.c8 = conv_block(
            base_c*4,
            base_c*2
        )

        self.up1 = nn.ConvTranspose2d(
            base_c*2,
            base_c,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionBlock(
            base_c,
            base_c,
            base_c//2
        )

        self.c9 = conv_block(
            base_c*2,
            base_c
        )

        self.out = nn.Conv2d(
            base_c,
            out_c,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        # Bottleneck
        c5 = self.c5(p4)

        # Decoder 4
        u4 = self.up4(c5)
        c4_att = self.att4(u4, c4)

        u4 = torch.cat(
            [u4, c4_att],
            dim=1
        )

        c6 = self.c6(u4)

        # Decoder 3
        u3 = self.up3(c6)
        c3_att = self.att3(u3, c3)

        u3 = torch.cat(
            [u3, c3_att],
            dim=1
        )

        c7 = self.c7(u3)

        # Decoder 2
        u2 = self.up2(c7)
        c2_att = self.att2(u2, c2)

        u2 = torch.cat(
            [u2, c2_att],
            dim=1
        )

        c8 = self.c8(u2)

        # Decoder 1
        u1 = self.up1(c8)
        c1_att = self.att1(u1, c1)

        u1 = torch.cat(
            [u1, c1_att],
            dim=1
        )

        c9 = self.c9(u1)

        return self.out(c9)

class HybridCBM(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(
            weights=None
        )

        feat_dim = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.encoder = backbone

        self.concept_head = nn.Sequential(

            nn.Linear(
                feat_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                len(CONCEPT_NAMES)
            ),

            
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                len(CONCEPT_NAMES),
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                len(CLASS_NAMES)
            ),
        )

    def forward(self,x):

        feat = self.encoder(x)

        concepts = self.concept_head(
            feat
        )

        logits = self.classifier(
            concepts
        )

        return concepts, logits

# ==========================================================
# CARE-MD
# ==========================================================

class CAREMD2:

    def __init__(self):

        # --------------------------------------
        # Device
        # --------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------
        # Models
        # --------------------------------------

        self.attunet = None

        self.hybrid = None

        # --------------------------------------
        # Memory
        # --------------------------------------

        self.memory_bank = None

        self.image_ids = None

        self.features = None

        # --------------------------------------
        # Current Case
        # --------------------------------------

        self.image_path = None

        self.image_id = None

        self.original = None

        self.mask = None

        self.roi = None

        self.feature = None

        self.clinical = None

        self.latent = None

        self.prediction = None

        self.confidence = None

        self.references = []

        self.gt = "Unknown"

        self.support_count = 0

        self.avg_similarity = 0

        self.validation = ""

        self.disease_distribution = {}

        self.phase3_prediction = None

        self.phase3_support = 0
        
        self.final_prediction = None
        
        self.decision_status = None

        # --------------------------------------
        # Initialization
        # --------------------------------------

        self._load_models()

        self._load_memory_bank()

        print("✓ CARE-MD Initialized")
    

    def _load_models(self):
    
            # Attention U-Net
            self.attunet = AttUNet().to(self.device)
    
            ckpt = torch.load(
                ATTUNET_MODEL,
                map_location=self.device,
                weights_only=False
            )
    
            self.attunet.load_state_dict(
                ckpt["model_state"]
            )
    
            self.attunet.eval()
    
            # Hybrid CBM
            self.hybrid = HybridCBM().to(self.device)
    
            ckpt = torch.load(
                HYBRID_MODEL,
                map_location=self.device,
                weights_only=False
            )
    
            self.hybrid.load_state_dict(
                ckpt["model_state"]
            )
    
            self.hybrid.eval()
    
            print("✓ Models Loaded")
        
        # -------------------------------------------------
    
    def _load_memory_bank(self):
    
            with open(
                MEMORY_BANK,
                "rb"
            ) as f:
    
                self.memory_bank = pickle.load(f)
    
            self.image_ids = list(self.memory_bank.keys())

            self.features = np.stack(
                [
                    self.memory_bank[k]["feature"]
                    for k in self.image_ids
                ]
            )
            
            print(f"✓ Memory Loaded : {len(self.image_ids)} cases")
                    

# ==========================================================
# PHASE-1 : ROI LOCALIZATION
# ==========================================================

    def phase1(self, image_path):
    
        self.image_path = image_path
    
        self.image_id = os.path.splitext(
            os.path.basename(image_path)
        )[0]
    
        # --------------------------------------
        # Read Original Image
        # --------------------------------------
    
        image = cv2.imread(image_path)
    
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    
        self.original = image.copy()
    
        h, w = image.shape[:2]
    
        # --------------------------------------
        # Model Input
        # --------------------------------------
    
        img = cv2.resize(
            image,
            (256,256)
        )
    
        img = img.astype(np.float32) / 255.0
    
        tensor = torch.tensor(
            img.transpose(2,0,1),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
    
        # --------------------------------------
        # Predict Mask
        # --------------------------------------
    
        with torch.no_grad():
    
            pred = self.attunet(tensor)
    
            pred = torch.sigmoid(pred)
    
            pred = pred.squeeze().cpu().numpy()
    
        mask = (pred > 0.5).astype(np.uint8)
    
        mask = cv2.resize(
            mask,
            (w,h),
            interpolation=cv2.INTER_NEAREST
        )
    
        self.mask = mask
    
        # --------------------------------------
        # Largest Connected Component
        # --------------------------------------
    
        contours, _ = cv2.findContours(
    
            mask,
    
            cv2.RETR_EXTERNAL,
    
            cv2.CHAIN_APPROX_SIMPLE
    
        )
    
        if len(contours) == 0:
    
            self.roi = self.original.copy()
    
        else:
    
            cnt = max(
                contours,
                key=cv2.contourArea
            )
    
            x,y,w_box,h_box = cv2.boundingRect(cnt)
    
            pad = 10
    
            x = max(0, x-pad)
    
            y = max(0, y-pad)
    
            x2 = min(
                self.original.shape[1],
                x+w_box+pad
            )
    
            y2 = min(
                self.original.shape[0],
                y+h_box+pad
            )
    
            self.roi = self.original[
                y:y2,
                x:x2
            ]
    
        # --------------------------------------
        # Save Outputs
        # --------------------------------------
    
        mask_path = os.path.join(
    
            MASK_DIR,
    
            self.image_id + "_mask.png"
    
        )
    
        roi_path = os.path.join(
    
            ROI_DIR,
    
            self.image_id + "_roi.jpg"
    
        )
    
        cv2.imwrite(
    
            mask_path,
    
            self.mask*255
    
        )
    
        cv2.imwrite(
    
            roi_path,
    
            cv2.cvtColor(
                self.roi,
                cv2.COLOR_RGB2BGR
            )
    
        )
    
        print("✓ Phase-1 Completed")
    
        print("Mask :", mask_path)
    
        print("ROI  :", roi_path)

# ==========================================================
# PHASE-2 : CONCEPT REASONING
# ==========================================================

    def phase2(self):
    
        # --------------------------------------
        # ROI → Model Input
        # --------------------------------------
    
        img = cv2.resize(
            self.roi,
            (224,224)
        )
    
        img = img.astype(np.float32) / 255.0
    
        tensor = torch.tensor(
            img.transpose(2,0,1),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
    
        # --------------------------------------
        # Forward Pass
        # --------------------------------------
    
        with torch.no_grad():
    
            feature = self.hybrid.encoder(tensor)
    
            concepts = self.hybrid.concept_head(feature)
    
            logits = self.hybrid.classifier(concepts)
    
            probs = torch.softmax(
                logits,
                dim=1
            )
    
            confidence, pred = torch.max(
                probs,
                dim=1
            )
    
        # --------------------------------------
        # Store Outputs
        # --------------------------------------
    
        self.feature = feature.squeeze().cpu().numpy()
    
        concepts = concepts.squeeze().cpu().numpy()
    
        self.clinical = concepts[:6]
    
        self.latent = concepts[6:]
    
        self.prediction = CLASS_NAMES[
            pred.item()
        ]
    
        self.confidence = float(
            confidence.item()
        )
    
        self.logits = logits.squeeze().cpu().numpy()
    
        print("✓ Phase-2 Completed")
    
        print("Prediction :", self.prediction)
    
        print(f"Confidence : {self.confidence:.4f}")

# ==========================================================
# PHASE-3 : CARE-REFERENCE
# ==========================================================

    def phase3(self, top_k=10):
    
        # --------------------------------------
        # Cosine Similarity
        # --------------------------------------
    
        similarity = cosine_similarity(
            self.feature.reshape(1, -1),
            self.features
        )[0]
    
        # --------------------------------------
        # Top-K
        # --------------------------------------
    
        top_idx = np.argsort(
            similarity
        )[::-1][:top_k]
    
        self.references = []
    
        for idx in top_idx:
    
            image_id = self.image_ids[idx]
    
            sample = self.memory_bank[image_id]
    
            self.references.append({
    
                "image_id": image_id,
    
                "dx": sample.get("dx", "Unknown"),
    
                "prediction": sample["prediction"],
    
                "confidence": sample["confidence"],
    
                "similarity": float(
                    similarity[idx]
                )
    
            })
    
        # --------------------------------------
        # Evidence Statistics
        # --------------------------------------
    
        self.support_count = sum(
    
            ref["prediction"] == self.prediction
    
            for ref in self.references
    
        )
    
        self.avg_similarity = np.mean(
    
            [r["similarity"] for r in self.references]
    
        )
    
        # Disease Distribution
    
        self.disease_distribution = {}
    
        for ref in self.references:
    
            dx = ref["prediction"]
    
            self.disease_distribution[dx] = (
    
                self.disease_distribution.get(dx,0)+1
    
            )
    
        # --------------------------------------
        # Validation
        # --------------------------------------
    
        if self.support_count >= 8:
    
            self.validation = "STRONGLY SUPPORTED"
    
        elif self.support_count >= 5:
    
            self.validation = "MODERATELY SUPPORTED"
    
        else:
    
            self.validation = "WEAKLY SUPPORTED"
    
        print("✓ Phase-3 Completed")
    
        print(
            f"Retrieved : {len(self.references)} references"
        )
    
        print(
            f"Support   : {self.support_count}/{top_k}"
        )
    
        print(
            f"Similarity: {self.avg_similarity:.4f}"
        )

    # --------------------------------------
# Phase-3 Final Prediction
# --------------------------------------

        preds = [
            ref["prediction"]
            for ref in self.references
        ]
        
        counter = Counter(preds)
        
        self.phase3_prediction = counter.most_common(1)[0][0]
        
        self.phase3_support = counter[self.phase3_prediction]
        
        # --------------------------------------
        # Final Decision
        # --------------------------------------
        
        if self.prediction == self.phase3_prediction:
        
            self.final_prediction = self.prediction
        
            self.decision_status = "CONSISTENT"
        
        else:
        
            self.final_prediction = self.phase3_prediction
        
            self.decision_status = "CONFLICT"

        print("Phase-2 :", self.prediction)
        print("Phase-3 :", self.phase3_prediction)
        print("Final   :", self.final_prediction)
        print("Status  :", self.decision_status)
# ==========================================================
# GENERATE CARE-MD REPORT
# ==========================================================

    def generate_report(self):
    
        report_path = os.path.join(
            REPORT_DIR,
            self.image_id + "_CAREMD_Report.png"
        )
    
        fig = plt.figure(
            figsize=(24,18),
            facecolor="#F8F9FA"
        )
    
        gs = fig.add_gridspec(
    
            nrows=5,
    
            ncols=5,
    
            height_ratios=[
                1.2,
                1.8,
                1.8,
                1.2,
                1.2
            ],
    
            hspace=0.65,
    
            wspace=0.35
    
        )

        fig.suptitle(
    
            "CARE-MD Clinical Decision Report",
    
            fontsize=22,
    
            fontweight="bold"
    
        )

        ax = fig.add_subplot(gs[0,0])
    
        ax.imshow(self.original)
    
        ax.set_title(
    
            "Original Image",
    
            fontsize=14,
    
            fontweight="bold"
    
        )
    
        ax.axis("off")

        ax = fig.add_subplot(gs[0,1])
    
        ax.imshow(self.roi)
    
        ax.set_title(
    
            "Phase-1 ROI",
    
            fontsize=14,
    
            fontweight="bold"
    
        )
    
        ax.axis("off")

        ax = fig.add_subplot(gs[0,2])
    
        ax.axis("off")
    
        diagnosis = [

            ["Image ID", self.image_id],
        
            ["Ground Truth", self.gt],
        
            ["Phase-2 Prediction", self.prediction.upper()],
        
            ["Phase-2 Confidence", f"{self.confidence*100:.2f}%"],
        
            ["Phase-3 Prediction", self.phase3_prediction.upper()],
        
            ["Evidence Support", f"{self.phase3_support}/10"],
        
            ["Final Decision", self.final_prediction.upper()],
        
            ["Decision Status", self.decision_status]
        
        ]
    
        table = ax.table(
    
            cellText=diagnosis,
    
            colLabels=[
    
                "Parameter",
    
                "Value"
    
            ],
    
            cellLoc="left",
    
            loc="center"
    
        )
    
        table.auto_set_font_size(False)
    
        table.set_fontsize(11)
    
        table.scale(1.2,2.0)
    
        ax.set_title(
    
            "Diagnosis Summary",
    
            fontsize=14,
    
            fontweight="bold"
    
        )

        ax = fig.add_subplot(gs[0,3:5])
    
        ax.axis("off")
    
        rows = []
    
        for name,val in zip(
    
            CONCEPT_NAMES,
    
            np.concatenate(
    
                [
    
                    self.clinical,
    
                    self.latent
    
                ]
    
            )
    
        ):
    
            rows.append(
    
                [
    
                    name,
    
                    f"{val:.3f}"
    
                ]
    
            )
    
        table = ax.table(
    
            cellText=rows,
    
            colLabels=[
    
                "Concept",
    
                "Value"
    
            ],
    
            cellLoc="left",
    
            loc="center"
    
        )
    
        table.auto_set_font_size(False)
    
        table.set_fontsize(10)
    
        table.scale(1.1,1.6)
    
        ax.set_title(
    
            "Concept Summary",
    
            fontsize=14,
    
            fontweight="bold"
    
        )

        fig.subplots_adjust(
            left=0.03,
            right=0.98,
            top=0.92,
            bottom=0.03,
            hspace=0.45,
            wspace=0.35
        )

        # ==========================================================
# TOP-10 REFERENCE GRID
# ==========================================================

        fig.text(
            0.5,
            0.66,
            "Top-10 Similar Clinical References",
            fontsize=16,
            fontweight="bold",
            ha="center"
        )
        
        for i, ref in enumerate(self.references):
        
            row = i // 5
            col = i % 5
        
            ax = fig.add_subplot(gs[1 + row, col])



            
            print("="*60)
            print("REFERENCE_IMAGE_DIR :", REFERENCE_IMAGE_DIR)
            print("Image ID            :", ref["image_id"])
            
            img_path = os.path.join(
                REFERENCE_IMAGE_DIR,
                ref["image_id"] + ".jpg"
            )
            
            print("Image Path          :", img_path)
            print("Exists?             :", os.path.exists(img_path))
            print("="*60)
            
            img = cv2.imread(img_path)


            
        
            if img is not None:
        
                img = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )
        
                ax.imshow(img)
        
            else:
        
                ax.text(
                    0.5,
                    0.5,
                    "Image\nNot Found",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red"
                )
        
            ax.set_xticks([])
            ax.set_yticks([])
        
            # Border Color
            if ref["prediction"] == self.prediction:
                color = "green"
            else:
                color = "red"
        
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        
            ax.set_title(
                f"{ref['image_id']}",
                fontsize=8
            )
        
            ax.text(
                0.5,
                -0.10,
                f"GT : {ref['dx']}\n"
                f"Pred : {ref['prediction']}\n"
                f"Conf : {ref['confidence']:.2f}\n"
                f"Sim : {ref['similarity']:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="black"
                )
            )

        # ==========================================================
# FINAL EXPLANATION
# ==========================================================

            ax = fig.add_subplot(gs[4, :])
            
            ax.axis("off")
            
            ax.set_title(
                "Clinical AI Explanation",
                fontsize=15,
                fontweight="bold"
            )
            
            ax.text(
                0.01,
                0.95,
                self.generate_explanation(),
                fontsize=11,
                va="top",
                wrap=True,
                bbox=dict(
                    facecolor="#F5F5F5",
                    edgecolor="black",
                    boxstyle="round,pad=0.5"
                )
            )

        plt.savefig(
            report_path,
            dpi=300,
            bbox_inches="tight"
        )
        
       
        
        print(f"✓ Report Saved : {report_path}")
        
        return report_path



# ==========================================================
# RUN COMPLETE PIPELINE
# ==========================================================

    def run(self, image_path):

        # Phase-1
        self.phase1(image_path)

        # Phase-2
        self.phase2()

        # Phase-3
        self.phase3()

        # Phase-4
        report_path = self.generate_report()

        return report_path

# ==========================================================
# FINAL EXPLANATION
# ==========================================================

    def generate_explanation(self):
    
        explanation = []
    
        explanation.append(
            f"The uploaded lesion was localized using the Phase-1 "
            f"Attention U-Net segmentation model."
        )
    
        explanation.append(
            f"The extracted ROI was analysed by the Hybrid Concept "
            f"Bottleneck Model."
        )
    
        explanation.append(
            f"The classifier predicted '{self.prediction.upper()}' "
            f"with a confidence of {self.confidence*100:.2f}%."
        )
    
        explanation.append(
            f"The CARE-Reference module retrieved "
            f"{len(self.references)} visually similar historical cases."
        )
    
        explanation.append(
            f"{self.phase3_support} of the {len(self.references)} "
            f"retrieved cases supported the diagnosis "
            f"'{self.phase3_prediction.upper()}'."
        )
    
        explanation.append(
            f"Average retrieval similarity was "
            f"{self.avg_similarity:.3f}."
        )
    
        if self.decision_status == "CONSISTENT":
    
            explanation.append(
                "The classifier prediction and retrieval evidence "
                "are consistent, increasing confidence in the final diagnosis."
            )
    
        else:
    
            explanation.append(
                "The classifier prediction and retrieval evidence "
                "are not fully consistent. Manual clinical review is recommended."
            )
    
        explanation.append(
            f"Final CARE-MD Decision: "
            f"{self.final_prediction.upper()} "
            f"({self.validation})."
        )
    
        return " ".join(explanation)
