# HeteroSync Learning (HSL): Addressing Data Heterogeneity in Distributed Medical Imaging

Data heterogeneity presents a significant challenge in distributed artificial intelligence (AI) for medical imaging, limiting model performance across diverse clinical settings. To address this, we propose HeteroSync Learning (HSL), a privacy-preserving distributed learning framework that mitigates data heterogeneity by aligning heterogeneous representation through: (1) the Shared Anchor Task (SAT), a homogeneous reference task that establishes cross-node representation alignment; and (2) a customized Auxiliary Learning Architecture that coordinates the co-optimization of SAT with local primary tasks. HSL is validated through large-scale simulations, covering feature, label, quantity, and combined heterogeneity scenarios, and applied to a real-world multi-center thyroid cancer diagnosis project. The results show that HSL outperforms local learning, four classical methods (Personalized Learning, FedAvg, FedProx, SplitAVG), eight state-of-the-art algorithms (e.g., FedRCL, FedCOME, FedDpS), and large foundation model (e.g., CLIP) by up to 40% in AUC, performing comparably to central learning. In generalization tests, HSL achieves an AUC of 0.846 on the out-of-distribution Pediatric Thyroid Cancer dataset, significantly outperforming other methods (AUC range: 0.564-0.795). Visualization results show that HSL transforms heterogeneous data distributions into homogeneous representations, demonstrating the effectiveness of its alignment mechanism. This study offers an efficient solution to the heterogeneity issue in distributed medical AI, promoting equitable collaboration among resource-unequal institutions, and advancing the democratization of healthcare AI.

This project code is owned by the MedAI Collaborative Laboratory. The code example is designed as a directly executable demo, with the following specifications:

The code example includes: 
- `Data` and `Results` folders
- Code files: `Demo.py`, `MMOE_ResNet18.py`, and `SSL_train.py`

## Project Structure

1. **Code Overview**  
   The example demonstrates a thyroid cancer diagnosis scenario with 3 nodes participating in distributed learning. This simplified example omits distributed communication and weight encryption/decryption code between nodes, focusing only on Node 1's training and testing process.

2. **Data Folder**  
   Contains sample datasets for training, validation, and testing:
   - Training set label distribution: Malignant:Benign = 7:7
   - Validation set: Malignant:Benign = 2:2  
   - Test set: Malignant:Benign = 1:1  
   *(Replace with your actual data)*

3. **Results Folder**  
   Stores model weights during training and validation/test results.  
   This folder also serves as the communication directory for distributed learning, where nodes download/upload model weights.  
   The example assumes weights from 2 other nodes have already been downloaded:
   - `Node1_dict_best.pkl`: Best weights from Node 1 during training
   - `Node2_dict_best.pkl`: Downloaded best weights from Node 2  
   - `Node3_dict_best.pkl`: Downloaded best weights from Node 3  
   - `final_dict_best.pkl`: Final global model after training

4. **Demo.py**  
   Simplified training-validation-testing code for Node 1:
   - `os.chdir('xx')`: Sets target directory (folder containing all example files)
   - `data_path='./Data'`, `save_dir='./Results'`: Paths to Data/Results folders
   - `dataset1_path=data_path+'/train'`: Main task dataset path  
   - `dataset2_path=data_path+'/Data/RSNA_LUNG/train'`: SAT dataset path  
   - `dataset_path=data_path+'/val'`: Validation dataset path during training  
   - `dataset_path=data_path+'/test'`: Final test dataset path  

5. **SSL_train.py**  
   Contains intermediate training/testing code called by Demo.py, including a balanced data sampling mechanism for datasets with significant label distribution differences.

6. **MMOE_ResNet18.py**  
   Core HSL model code combining MMOE with ResNet, incorporating a temperature parameter "T" (based on knowledge distillation principles).

7. **Training Notes**  
   Adjust `batch_size` and `num_workers` according to your GPU/CPU configuration.

For questions or suggestions, please contact us.
