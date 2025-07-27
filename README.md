# Drosophila Grooming Analysis Pipeline - Specifications

This repository contains the official design documents and specifications for a three-stage pipeline to analyze *Drosophila* grooming behavior from time-lapse microscopy videos.

The primary goal of this project is to create an optimized manual sampling strategy that maintains statistical power while dramatically reducing analysis time.

## Key Components
The proposed solution is a pipeline that:
1.  **Determines optimal sampling parameters** from pilot data.
2.  **Generates sampled videos** for manual scoring.
3.  **Performs comprehensive statistical analysis** comparing genotypes, with robust handling for edge events and multiple testing.

This specification builds upon the existing `wilya7/fly_behavior_analysis` repository, reusing its core data validation and processing functionalities.

**➡️ For the full, detailed specification, please see [DESIGN.md](DESIGN.md).**