# title: 
'md-graph-converter: A Python Framework for Building Atom-Level Graphs from Heterogeneous MD Trajectories'

# tags:
  . Python

  . molecular dynamics
  
  . graph neural networks
  
  . computational biology
  
  . membrane proteins

# authors:
  - name: Fodil Azzaz

    orcid: 0000-0002-2508-2300
    
    affiliation: 1
    
affiliations:

  - name: Aix-Marseille Univ, INSERM UA16, Marseille, France


    index: 1
    
date: 11 March 2026

bibliography: paper.bib
---

# Abstract
Molecular dynamics (MD) simulations provide atomistic descriptions of biomolecular systems with high spatial and temporal resolution. However, translating these data into representations suitable for modern machine learning approaches—particularly graph-based and geometric deep learning models—remains a significant challenge, especially for heterogeneous systems such as membrane proteins embedded in complex lipid, solvent, and ionic environments. Here, we present a framework for constructing atom-level graphs directly from MD trajectories using PSF topology and DCD coordinate files. Each atom is represented as a node enriched with physicochemical and structural features, including charge, mass, atomic identity, residue hydrophobicity, and backbone or side-chain context. Edges encode both covalent connectivity and non-covalent interactions through distance-based radial basis functions and directional vectors, enabling compatibility with equivariant neural architectures. To balance biological relevance and computational efficiency, the framework allows selective inclusion of intra- and/or inter-molecular non-covalent interactions. The method is applicable to diverse biomolecular systems, including soluble proteins, membrane proteins, lipids, cholesterol, gangliosides, water, and ions, with user-defined segment identification. This work provides a flexible and physically informed bridge between molecular dynamics simulations and graph-based machine learning, facilitating the analysis and modeling of complex biological systems.


# Introduction

Molecular dynamics (MD) simulations provide atomistic insight into the structure and dynamics of biomolecular systems. With advances in force fields and computational power, simulations now routinely include heterogeneous environments such as membrane proteins embedded in lipid bilayers and solvated by explicit water and ions 1,2. Despite this progress, converting MD trajectories into representations suitable for modern machine learning—particularly graph-based and geometric deep learning models—remains nontrivial.


Graph representations offer a natural abstraction of molecular systems, where atoms or residues are encoded as nodes and their interactions as edges. However, most existing graph construction pipelines focus on isolated proteins or small molecules and frequently neglect the surrounding molecular environment. This omission is especially limiting for membrane proteins, whose structure and function depend strongly on lipid composition3. Furthermore, many workflows rely solely on PDB files, which may lack explicit bonding topology and require reconstruction steps that introduce ambiguity4. In practice, most coordinate-based graph construction approaches rely on heuristic bond inference or purely distance-based radius graphs, often neglecting explicit force-field topology, segment identity, and heterogeneous environmental composition intrinsic to production MD simulations.


Here, we present a generalized framework for constructing atom-level graphs directly from PSF topology files and DCD trajectories, formats commonly used in large-scale MD simulations with NAMD1 and CHARMM-based force fields2. The method preserves explicit covalent connectivity, extracts non-covalent interactions from dynamic coordinates, and annotates each atom with physicochemical and structural features. Designed to operate with user-defined segment identification, the framework supports heterogeneous systems including proteins, lipids, solvent, and ions. Optional restriction to inter-molecular interactions allows efficient graph generation for large membrane simulations. This framework establishes a topology-consistent and reproducible infrastructure layer that systematically bridges production-scale molecular dynamics simulations with geometric deep learning architectures.

# Methods

Graph Definition. Each molecular system is represented as a graph in which atoms correspond to nodes and their interactions correspond to edges. Graphs are constructed directly from PSF topology files and DCD trajectory files, thereby preserving explicit bonding information together with atomic coordinates obtained from molecular dynamics simulations. Each atom is associated with a feature vector describing its physicochemical and structural properties, and each edge is associated with features characterizing geometric relationships and bonding type between atom pairs. Graphs can be generated for individual frames of a trajectory, enabling time-resolved representation of dynamic molecular systems. By directly leveraging PSF topology rather than reconstructing connectivity from coordinates, the covalent structure remains fully consistent with the underlying force field throughout graph generation.
Input Parsing and System Representation. Topology information is extracted from the PSF file, including atomic identity, partial charges, atomic masses, covalent bond connectivity, and segment identifiers. Atomic coordinates are read from DCD trajectory files using MDAnalysis5, allowing direct processing of simulations performed with NAMD and CHARMM-based force fields. Because bonding information is obtained directly from the topology file, the covalent structure of the system is preserved without requiring reconstruction from coordinate data alone. The framework is designed to operate with user-defined segment identification rather than hard-coded residue names or predefined molecule categories. Molecular group identity is inferred from topology and segment identifiers, with a default "OTHER" category ensuring robust handling of unexpected components. This design enables application to soluble proteins, membrane proteins, lipid bilayers, cholesterol, gangliosides, water molecules, and ions within a unified and system-agnostic workflow.


Node Feature Construction. Each atom is encoded as a node enriched with physicochemically meaningful features. The feature vector includes partial atomic charge and atomic mass obtained from the PSF topology, atomic number inferred from element identity, normalized hydrophobicity based on the Kyte–Doolittle scale6, residue charge classification as positive, negative, or neutral, backbone and side-chain indicators to capture structural context, and one-hot encoding of user-defined segment identifiers. This feature design integrates atomic-level and residue-level information in a representation suitable for graph-based machine learning models7,8.


Edge Construction. Edges are constructed from two complementary sources: explicit covalent bonds and distance-based non-covalent interactions. Covalent connectivity is extracted directly from the PSF topology file, and an undirected edge is created for each bonded atom pair. These edges are labeled to indicate covalent bonding, thereby preserving the force-field-defined molecular structure. Non-covalent edges are generated by applying a user-defined distance cutoff to atomic coordinates within each trajectory frame. For computational efficiency, neighbor search is implemented using KD-tree algorithms. Atom pairs within the specified cutoff are connected by an edge representing a potential non-covalent interaction. To balance computational efficiency with biological relevance, the framework allows optional restriction of non-covalent edges to inter-molecular interactions only. This controlled sparsification is particularly relevant for large heterogeneous membrane systems, where full distance-based graphs can become prohibitively dense and obscure biologically meaningful interface interactions. In this mode, intra-molecular pairs are excluded from distance-based edge construction, reducing graph density in large systems such as membrane simulations while preserving biologically meaningful intermolecular contacts.

Edge Feature Encoding. Each edge is associated with scalar and vector features that describe the geometric and chemical relationship between connected atoms. Interatomic distance is encoded using a radial basis function expansion, providing a smooth representation of spatial separation suitable for neural network learning. A binary indicator specifies whether an edge corresponds to a covalent bond or a non-covalent interaction. A normalized direction vector is computed for each atom pair to capture orientation in three-dimensional space. This orientation-aware representation enables compatibility with geometric and equivariant graph neural network architectures9,10.

Computational Considerations. Graph construction scales with system size and the selected distance cutoff for non-covalent interactions. Restricting non-covalent edges to inter-molecular interactions significantly reduces computational cost and memory usage in large heterogeneous systems. Because each trajectory frame can be processed independently, the workflow supports parallelization across time steps. The resulting graph objects are exported in machine-learning-ready formats compatible with PyTorch Geometric11 and similar libraries, enabling seamless integration into downstream modeling pipelines.

# Usage Example

To illustrate the versatility of the graph-generation framework, a representative system containing Botulinum Neurotoxin B (BoNT/B), its receptor Synaptotagmin 2 (SYT2) , and a ganglioside was constructed12,13. The generated graph captures atomic-level details of each component, including backbone and side-chain atoms of the protein, as well as the sugar and fatty acid chains of the ganglioside. Each node carries physicochemical and structural features, and edges represent covalent bonds and non-covalent interactions, as described in Table 1.

For visualization purposes, nodes are colored by molecular segment: BoNT/B in green, SYT2 in blue, and the ganglioside in orange. 


The framework provides flexible user control over graph construction. Users can select which atoms to include for the protein and the environment, define the cutoff distance for non-covalent interactions, specify the number and spacing of trajectory frames, and define their own segment identifiers.
This example demonstrates that even with a minimal system of two proteins and a ganglioside, the framework produces a complete atomic graph representation. Users can easily adapt the pipeline to their own systems, maintaining full control over computational resources and graph content, while capturing the spatial and chemical organization of a biologically meaningful molecular interface.
Discussion and conclusion


The framework presented here formalizes a standardized and topology-consistent workflow for transforming heterogeneous molecular dynamics simulations into graph representations suitable for geometric deep learning. By preserving force-field-defined covalent connectivity, explicit environmental composition, and physicochemically meaningful node and edge attributes, the method provides a biologically faithful abstraction of complex molecular assemblies.


Its adaptability is illustrated by the example system of BoNT/B, SYT2, and a ganglioside, but the same methodology can be applied to any protein, ligand, or membrane environment compatible with NAMD/CHARMM36m simulations. Users have full control over graph construction, including atom selection for proteins and environment, cutoff distances for non-covalent interactions, inclusion of inter- or intra-molecular edges, number and spacing of trajectory frames, and segment definitions. This flexibility allows the framework to be tailored to specific biological questions while balancing computational efficiency.
Several design choices distinguish this approach from existing pipelines. Charges and atomic masses are extracted directly from PSF topology files, preserving physical fidelity from the force field. User-defined segment encoding enables robust handling of heterogeneous systems, including proteins, lipids, solvent, and ions, without relying on hard-coded residue lists. Optional restriction to inter-molecular non-covalent interactions reduces graph density for large systems while retaining biologically meaningful interactions.


This framework has already been applied in a broader context in our Perturbation Scanning (PS) study14, where graphs generated with this tool served as the foundation for a deep learning pipeline to dissect residue-level molecular interactions. This demonstrates that the tool is not only versatile in principle but also practically applicable for advanced computational analyses.
Importantly, the contribution of this work lies not in introducing a new graph theory formulation, but in establishing a reproducible and physically grounded conversion infrastructure that enables systematic integration of MD simulations with modern graph-based AI models.
Overall, this generalized, user-configurable graph-construction framework bridges molecular dynamics simulations with graph-based AI models, offering a computationally tractable and biologically meaningful representation of complex biomolecular systems. Its utility spans structural characterization, machine learning applications, and rational molecular design, highlighting its potential for broad impact across computational biology, biophysics, drug discovery, and protein engineering.
Limitations and Future Directions
The current implementation of md-graph-converter has several limitations that define the scope of its applicability and suggest avenues for subsequent development. First, the framework is specifically designed for the PSF topology and DCD trajectory formats native to NAMD and CHARMM force fields. While this ensures high fidelity for users of these simulation packages, it currently precludes direct support for other major MD engines such as GROMACS, AMBER, or OpenMM without intermediate format conversion. Second, the detection of non-covalent interactions relies on a simple, user-defined distance cutoff. Although computationally efficient, this geometric criterion may lack the physical specificity of energy-based or more sophisticated statistical approaches (e.g., persistent homology) for defining persistent or biologically relevant contacts. Finally, graph construction scales with system size and the selected cutoff; for very large systems or long trajectories, memory usage can become substantial, potentially requiring strategic frame selection or the use of high-memory computing resources.

# Code Availability
The Python framework for constructing atom-level graphs from molecular dynamics trajectories is freely available on GitHub: https://github.com/fodil13/md-graph-converter. The repository includes installation instructions, and tutorials to facilitate graph generation for diverse biomolecular systems.

# Author Contribution
F.A.: Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing – Original Draft, Writing – Review & Editing, Visualization

# Competing Interests
F.A is the creator of the md-graph-convert software, which is made available under a custom open-source license.

# References
(1)	Phillips, J. C.; Braun, R.; Wang, W.; Gumbart, J.; Tajkhorshid, E.; Villa, E.; Chipot, C.; Skeel, R. D.; Kalé, L.; Schulten, K. Scalable Molecular Dynamics with NAMD. J. Comput. Chem. 2005, 26 (16), 1781–1802. https://doi.org/10.1002/jcc.20289.

(2)	Brooks, B. R.; Brooks III, C. L.; Mackerell Jr., A. D.; Nilsson, L.; Petrella, R. J.; Roux, B.; Won, Y.; Archontis, G.; Bartels, C.; Boresch, S.; Caflisch, A.; Caves, L.; Cui, Q.; Dinner, A. R.; Feig, M.; Fischer, S.; Gao, J.; Hodoscek, M.; Im, W.; Kuczera, K.; Lazaridis, T.; Ma, J.; Ovchinnikov, V.; Paci, E.; Pastor, R. W.; Post, C. B.; Pu, J. Z.; Schaefer, M.; Tidor, B.; Venable, R. M.; Woodcock, H. L.; Wu, X.; Yang, W.; York, D. M.; Karplus, M. CHARMM: The Biomolecular Simulation Program. J. Comput. Chem. 2009, 30 (10), 1545–1614. https://doi.org/10.1002/jcc.21287.

(3)	Azzaz, F.; Mazzarino, M.; Chahinian, H.; Yahi, N.; Scala, C. D.; Fantini, J. Structure of the Myelin Sheath Proteolipid Plasmolipin (PLLP) in a Ganglioside-Containing Lipid Raft. Front. Biosci. Landmark Ed. 2023, 28 (8), 157. https://doi.org/10.31083/j.fbl2808157.

(4)	Berman, H. M.; Westbrook, J.; Feng, Z.; Gilliland, G.; Bhat, T. N.; Weissig, H.; Shindyalov, I. N.; Bourne, P. E. The Protein Data Bank. Nucleic Acids Res. 2000, 28 (1), 235–242. https://doi.org/10.1093/nar/28.1.235.

(5)	Michaud-Agrawal, N.; Denning, E. J.; Woolf, T. B.; Beckstein, O. MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations. J. Comput. Chem. 2011, 32 (10), 2319–2327. https://doi.org/10.1002/jcc.21787.

(6)	Kyte, J.; Doolittle, R. F. A Simple Method for Displaying the Hydropathic Character of a Protein. J. Mol. Biol. 1982, 157 (1), 105–132. https://doi.org/10.1016/0022-2836(82)90515-0.

(7)	Kipf, T. N.; Welling, M. Semi-Supervised Classification with Graph Convolutional Networks. CoRR 2016, abs/1609.02907.

(8)	Battaglia, P. W.; Hamrick, J. B.; Bapst, V.; Sanchez-Gonzalez, A.; Zambaldi, V.; Malinowski, M.; Tacchetti, A.; Raposo, D.; Santoro, A.; Faulkner, R.; Gulcehre, C.; Song, F.; Ballard, A.; Gilmer, J.; Dahl, G.; Vaswani, A.; Allen, K.; Nash, C.; Langston, V.; Dyer, C.; Heess, N.; Wierstra, D.; Kohli, P.; Botvinick, M.; Vinyals, O.; Li, Y.; Pascanu, R. Relational Inductive Biases, Deep Learning, and Graph Networks, 2018. https://arxiv.org/abs/1806.01261.

(9)	Fuchs, F. B.; Worrall, D. E.; Fischer, V.; Welling, M. SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks, 2020. https://arxiv.org/abs/2006.10503.

(10)	Liao, Y.-L.; Smidt, T. Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs, 2023. https://arxiv.org/abs/2206.11990.

(11)	Fey, M.; Lenssen, J. E. Fast Graph Representation Learning with PyTorch Geometric, 2019. https://arxiv.org/abs/1903.02428.

(12)	Azzaz, F.; El Far, O.; Fantini, J. Membrane Constraints Reshape Synaptotagmin Recognition by Botulinum Neurotoxin B1. bioRxiv 2025. https://doi.org/10.1101/2025.08.27.672333.

(13)	Ramirez-Franco, J.; Azzaz, F.; Sangiardi, M.; Ferracci, G.; Youssouf, F.; Popoff, M. R.; Seagar, M.; Lévêque, C.; Fantini, J.; El Far, O. Molecular Landscape of BoNT/B Bound to a Membrane-Inserted Synaptotagmin/Ganglioside Complex. Cell. Mol. Life Sci. CMLS 2022, 79 (9), 496. https://doi.org/10.1007/s00018-022-04527-4.

(14)	Azzaz, F.; Fantini, J. An AI-Driven Platform for Deconstructing and Engineering Biomolecular Recognition. bioRxiv 2025. https://doi.org/10.64898/2025.12.09.692808.


Table 1: Node and Edge Features for Molecular Graphs

| Feature Type | Name / Description | Details / Representation |
|-------------|-------------------|-------------------------|
| Node | Atom identity | Derived from topology (residue name + atom name) |
| Node | Partial charge | Taken from PSF topology; normalized for ML stability |
| Node | Atomic mass | From PSF; normalized |
| Node | Atomic number | Inferred from element |
| Node | Residue hydrophobicity | Normalized Kyte-Doolittle scale |
| Node | Residue charge category | Positive, negative, or neutral |
| Node | Backbone indicator | Flag distinguishing backbone atoms |
| Node | Side-chain indicator | Flag distinguishing side-chain atoms |
| Node | Segment identifier | One-hot encoding of user-defined molecular group |
| Edge | Covalent connectivity | Explicit bond from PSF topology; labeled as covalent |
| Edge | Non-covalent interactions | Distance-based edges; optional inter-molecular only |
| Edge | Distance encoding | Radial basis function expansion of interatomic distance |
| Edge | Bond type indicator | Binary: covalent vs non-covalent |
| Edge | Direction vector | Normalized vector pointing from source to target atom |


 
Figure 1. Atomic-level graph representation and customizable pipeline for a minimal system. (A) Node-only visualization of BoNT/B (green), SYT2 (blue), and a ganglioside (orange). Each node encodes features including atom type, charge, hydrophobicity, and residue information as described in Table 1. Edges exist in the graph data but are omitted for clarity. (B) Example snippet of the Python code illustrating how users can customize protein and environment selection, number of frames, frame spacing, and non-covalent edge computation when generating graphs. (figure1.PNG)






