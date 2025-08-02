# Hydrogen Storage Materials Data Extraction and Analysis Project

## Overview

This project is an automated system for extracting and analyzing hydrogen storage material data from scientific literature. It combines natural language processing, computer vision, and machine learning to process research papers and extract structured information about hydrogen storage materials, their properties, and performance characteristics.

## Key Features

- **Literature Data Extraction**: Automated extraction of material properties from scientific papers using Large Language Models (LLMs)
- **Multi-modal Analysis**: Processing both text and image data from research publications
- **Material Classification**: Systematic categorization of hydrogen storage materials into different types
- **Machine Learning Modeling**: Predictive modeling for material properties using extracted data
- **Workflow Automation**: End-to-end pipeline for processing large datasets of scientific papers

## Project Components

### 1. DIVE_workflow.py
Main workflow orchestration script that handles:
- Paper processing pipeline using LangGraph state management
- Multi-step data extraction from scientific publications
- Integration with various LLM providers (OpenAI, Anthropic, Google, etc.)
- Parallel processing of multiple papers
- Image and text analysis for comprehensive data extraction
- Support for different material types and extraction schemas

Key capabilities:
- Two-step extraction process for improved accuracy
- Configurable LLM models and parameters
- Concurrent processing with customizable worker limits
- Automatic saving and progress tracking
- Support for multiple data sources and formats

### 2. prompt_template.py
Comprehensive prompt engineering module containing:
- Material type classification schemas for hydrogen storage materials:
  * Interstitial Hydrides (AB2, AB3, AB5 categories)
  * Complex Hydrides
  * Ionic Hydrides
  * Porous Materials (MOFs, COFs, carbon-based)
  * Multi-component Hydrides
  * Superhydrides
- Structured output parsers for consistent data extraction
- Specialized prompts for different data types:
  * Text-only extraction
  * Image analysis (PCT curves, discharge curves, TPD data)
  * Combined text-image processing
- Data reformatting and standardization templates

### 3. ml_modeling.py
Machine learning pipeline for predictive modeling:
- Feature engineering using composition-based descriptors
- Integration with matminer for materials informatics features
- XGBoost regression for property prediction
- Automated data preprocessing and cleaning
- Element fraction calculations for compositional analysis
- Model training and evaluation workflows

## Material Categories Supported

The system can identify and classify various hydrogen storage materials:

1. **Interstitial Hydrides**: TiFe, LaNi₅, ZrV₂, TiMn₂, etc.
2. **Complex Hydrides**: NaAlH₄, LiAlH₄, LiBH₄, Mg(BH₄)₂, etc.
3. **Ionic Hydrides**: LiH, NaH, MgH₂, CaH₂, etc.
4. **Porous Materials**: MOFs, COFs, activated carbon, graphene, etc.
5. **Multi-component Hydrides**: Composite materials and mixtures
6. **Superhydrides**: High-pressure hydrides like LaH₁₀, YH₉, etc.

## Data Extraction Capabilities

The system extracts comprehensive material properties including:
- Chemical formulas and compositions
- Hydrogenation/dehydrogenation conditions (temperature, pressure)
- Hydrogen storage capacities (gravimetric and volumetric)
- Thermodynamic properties (enthalpy, entropy changes)
- Kinetic data from experimental measurements
- Performance metrics from various characterization techniques

## Technical Requirements

### Dependencies
- Python 3.8+
- LangChain and LangGraph for LLM orchestration
- OpenAI, Anthropic, Google AI APIs
- Scikit-learn and XGBoost for machine learning
- Matminer and PyMatGen for materials informatics
- Pandas and NumPy for data processing
- Various image processing libraries

### Configuration
The system supports multiple LLM providers and can be configured for:
- Different model types and sizes
- Custom API endpoints
- Adjustable processing parameters
- Flexible output formats

## Usage

1. **Setup**: Configure API keys and model endpoints in the configuration section
2. **Data Preparation**: Organize input papers in the specified directory structure
3. **Extraction**: Run the DIVE workflow to process papers and extract data
4. **Analysis**: Use the extracted data for machine learning modeling and analysis
5. **Validation**: Review and validate extracted information against ground truth data

## Input/Output

- **Input**: Scientific papers in PDF format (processed via MinerU)
- **Processing**: Multi-modal extraction from text and images
- **Output**: Structured CSV files with comprehensive material data

## Applications

This system is designed for:
- Materials discovery and screening
- Database construction for hydrogen storage materials
- Property prediction and optimization
- Literature review automation
- Research trend analysis

## Development Status 

This project represents an active research and development effort in automated materials informatics, combining state-of-the-art AI techniques with domain expertise in hydrogen storage materials.

## Note

This system requires appropriate API keys and access to LLM services. Ensure proper configuration of environment variables and model endpoints before use.
