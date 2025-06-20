# MAFI-DTA: A Multi-Modal Attention Framework Integrating Molecular Subgraphs and Homologous Sequence Features for Drug-Target Affinity Prediction


## 💡 MAFI-DTA Framework
MAFI: A Multi-Modal Attention Framework that combines protein homology sequence generation with multi-scale protein graph construction to enhance Drug-Target Affinity prediction. It uses the ESM3 model for creating homology sequences, enriching protein sequence diversity perception and improving generalization. The multi-step protein subgraph mapping captures structural info at different scales. Along with BILSTM and attention mechanisms, MAFI offers high interpretability, making it a credible tool for drug discovery.



 
## 🧠 File list
vocabulary_builder.py: Used to build a vocabulary, containing the TorchVocab, Vocab, and WordVocab classes, as well as the main function to generate and save the vocabulary.

utils.py: Provides a set of utility functions, including model training, prediction, and performance metric calculation.

molecular_interaction.py: Defines the MolecularInteractionDataset class for processing molecular and protein interaction datasets.

main.py: The main program file, containing the workflow for data preprocessing, model training, and evaluation.

interaction_network.py: Defines the model architecture, including the SpatialFeatureAggregator, HeterogeneousAttentionLayer, MolecularGraphNetwork, and CrossModalInteractionNet classes.

graphmaker.py: Provides functions for constructing molecular and protein contact graphs.

generate_homologous.py: Contains functions for generating homologous protein sequences and related utility functions.

These files collectively form a multi-modal attention framework for drug-target affinity prediction.


## 📁 Dataset

### Proteins, Small molecules, and Affinity values
To ensure the accuracy and generalisation capability of the model, this study used six authoritative datasets: Davis, KIBA, PDBbind, Toxcast, Binding DB, and Metz. The following is a brief description of each dataset and the link to obtain it:

1. Davis Dataset: Records protein-drug molecular binding affinity data for model training and validation. This dataset provides rich protein-drug interaction information, aiding the model in learning predictive patterns for binding affinity. It can be downloaded via the following link: [Davis Dataset](https://davischallenge.org/).

2. KIBA Dataset: Integrates multi-source data to record ligand-receptor protein binding constants. This dataset provides comprehensive drug-target interaction data for models by combining information from various types of biological activity. Download link: [KIBA Dataset](https://paperswithcode.com/dataset/kiba).

3. PDBbind Dataset: Provides three-dimensional structures and binding affinity data for protein-ligand complexes. The three-dimensional structural information in this dataset helps models understand protein-ligand binding patterns. Download link: [PDBbind Dataset](https://www.bindingdb.org/bind/).

4. Toxcast Dataset: Covers toxicological data for various chemicals. This dataset provides models with rich toxicological information, aiding in toxicity prediction tasks. It can be accessed via the following link: [Toxcast Dataset](https://www.epa.gov/chemical-research).

5. Binding DB Dataset: Contains binding data for protein-ligand complexes. This dataset provides a large number of binding affinity measurements, offering abundant data support for model training. The download link is: [Binding DB Dataset](https://www.bindingdb.org/rwd/bind/index.jsp).

6. Metz Dataset: Used for toxicity prediction studies of compounds. This dataset focuses on toxicity assessment of compounds, providing critical data support for toxicity prediction tasks. It can be downloaded via the following link: [Metz Dataset](https://www.selectdataset.com/dataset/).

Users can download the datasets based on their characteristics and research needs, perform corresponding preprocessing operations, and fully leverage the advantages of these datasets to enhance model performance and generalisation capabilities.

### Protein Contact Map

Before starting model training, we use the ESM3 model to predict contact maps. The ESM3 model can predict the spatial proximity relationships between atoms in protein-ligand complexes based on protein sequence information, thereby generating corresponding contact maps. These contact maps include important details such as contact locations, contact types, and contact strengths, providing rich structural feature data for model training and helping the model better understand the interaction patterns between proteins and ligands.
Protein contact maps are named as ‘{target_key}.npy’ files. Each contact map for protein-ligand complexes in the generated dataset should be properly stored in a folder named ‘{dataset_name}_npy_contact_maps’ to enable quick and accurate access and reading during model training, ensuring the efficiency and consistency of the entire model training process and laying a solid foundation for improving model performance.

## ✨ Operating System
MAFI was developed on a Linux environment with CUDA 12.4

Hardware: Two NVIDIA GeForce RTX 4090（24G）

## 🛠️ Environment Setup
You'll need to run the following commands in order to run the codes
```
conda env create -f requirements.yml   # Create environment and install dependencies
```
it will download all the required libraries。Of course, you can also download it manually, but this is not recommended here.
```
conda activate mafi  # Activate environment mafi
```

## 🔗 Install ESM Model

If you want to install the ESM model, you can use the following command:

```
pip install fair-esm
```

```
pip install git+https://github.com/facebookresearch/esm.git
```

If you need to use a specific version of the ESM model, it will be automatically downloaded when you run the code for the first time. If the automatic download fails, you can manually download the model file from the official ESM model repository and place it in the appropriate directory.This section is mainly used to generate protein contact diagrams for easy input into the model. Of course, you can also use the provided protein contact diagram files.

Next,you need to follow the same steps to download the corresponding esm3 weight file in 

```
https://hf-mirror.com/EvolutionaryScale/esm3-sm-open-v1/tree/main/data
```
This allows you to use it to generate protein homologous sequences.


## 🖥️ Run Code
After thorough preparation, the code file can now be run.
```
python main.py 
```

## ✉ Citation and  Contact
@article{
}
