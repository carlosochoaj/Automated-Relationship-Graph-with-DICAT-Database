Collecting workspace information# Automated-Relationship-Graph-with-DICAT-Database

## Automated Relationship Graph for 16th–18th Century Theatre Professionals Using the DICAT Database

### Project Overview

This project aims to automatically create relationship graphs for theatre professionals from 16th-18th century Spain, using data from the DICAT database (https://dicat.uv.es/consulta/busqueda). The DICAT database contains descriptive texts about theatre professionals, including information about their relationships (e.g., working in the same theatre company, family relationships, etc.) in natural language format.

The challenge lies in processing these natural language texts to extract structured relationship information that can be visualized as a graph network of connections between theatre professionals.

### Repository Structure

- DB/ - Contains the DICAT database files in CSV format
  - Cleaned and original data for Juan Rana case study
  
- Supervised Models/ - Implementation of machine learning approaches
  - LSR/ - Language-structure recognition model
  - Rebel Model/ - Relationship extraction using REBEL architecture
    - Outputs relationship triplets in CSV format
    - Visualizations of the extracted network

- Symbolic/ - Rule-based approaches to relationship extraction
  
- Prompting/ - LLM-based approaches using prompt engineering


### Methodology

We explore three different approaches to solve the relationship extraction problem:

1. **Supervised Models** - Fine-tuning pre-trained language models (like mREBEL) on relationship extraction tasks
2. **Symbolic Solutions** - Using rule-based systems to identify patterns in the text
3. **Prompting** - Leveraging LLMs with carefully designed prompts to extract relationships

### Installation and Usage

1. Clone this repository
```bash
git clone https://github.com/yourusername/Automated-Relationship-Graph-with-DICAT-Database.git
cd Automated-Relationship-Graph-with-DICAT-Database
```

2. Install the required dependencies (detailed in each approach's directory)

3. Run the Jupyter notebooks to process the data and generate visualizations:
   - rebel_model.ipynb for the REBEL approach
   - LSR.ipynb for the Language Structure Recognition approach

4. View the resulting visualizations in the visualizations directory

### Results

The project has generated several visualizations of the relationship networks:
- Network graphs showing connections between theatre professionals
- Chord diagrams highlighting relationship types and frequencies
- Comparative visualizations between different extraction approaches

### Case Study: Juan Rana

We've used Juan Rana's data as a case study to evaluate our approaches, with the processed data available in the DB/ directory.

