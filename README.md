# UrbanIndex: Urban Morphometric Parameter Calculation for WRF Modeling

UrbanIndex is a computational tool for calculating urban morphometric parameters required by the Weather Research and Forecasting (WRF) model's urban canopy schemes. The software processes OpenStreetMap (OSM) data to derive spatially-explicit urban parameters on regular grids, including building heights, urban fractions, roughness lengths, and aerodynamic properties.

## Overview

The tool implements a Smart Buffer methodology for adaptive boundary detection, automatically identifying actual urban extent using density-based analysis. This approach provides computational efficiency improvements of 20-80% compared to traditional fixed municipal boundary methods, while maintaining accuracy in parameter estimation.

## Key Features

**Smart Buffer Methodology**
- Adaptive boundary detection using density-based urban extent identification
- Dynamic parameter adjustment based on city size and morphological characteristics
- Computational optimization through FRC_URB2D threshold processing

**Dual Processing Methods**
- Smart Buffer: Density-based urban extent detection with adaptive parameters
- Easy Boundary: Administrative boundary method using OSMnx for rapid analysis

**Technical Capabilities**
- Parallel processing with multi-core computation support
- Unified feature processing with geometric overlap resolution
- Quality control and validation with comprehensive error checking
- Direct WRF integration through CSV output compatibility

## Installation

## Installation

```bash
git clone https://github.com/NicMan89/Urban-Index-for-WRF.git
cd urbanindex
pip install -r requirements.txt
```

## Usage

### Command Line Interface

**Smart Buffer Method (default)**:
```bash
python UrbanIndex_v3.py --city "Rome, Italy" --verbose
```

**Easy Boundary Method**:
```bash  
python UrbanIndex_v3.py --city "Milan, Italy" --easy-boundary --verbose
```

**Custom Parameters**:
```bash
python UrbanIndex_v3.py --city "Naples, Italy" --grid 250 --threshold 0.15 --buffer 400
```

### Python API
```python
from UrbanIndex_v3 import calculate_urban_parameters_smart_buffer

workflow = calculate_urban_parameters_smart_buffer(
    city_name="Florence, Italy",
    grid_size_m=500,
    frc_threshold=0.20,
    use_easy_boundary=False
)

if workflow:
    workflow.print_statistics()
    workflow.plot_results('FRC_URB2D')
```

## Parameters

| Parameter | Description | Units | WRF Application |
|-----------|-------------|-------|------------------|
| FRC_URB2D | Urban fraction (impervious surfaces) | [-] | Urban/non-urban classification |
| BLDFR_URB2D | Plan area index (building footprints) | [-] | Building density indicator |
| MH_URB2D | Mean building height | [m] | Urban canopy height |
| H2W | Height-to-width ratio | [-] | Urban canyon characterization |
| BW | Building width | [m] | Geometric parameter |
| Z0 | Roughness length | [m] | Wind profile calculations |
| ZD | Zero-plane displacement height | [m] | Atmospheric boundary layer |
| Sigma | Height standard deviation | [m] | Urban heterogeneity |

## Command Line Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--city` | string | City name (required) | - |
| `--grid` | integer | Grid cell size in meters | 1000 |
| `--threshold` | float | FRC_URB2D threshold | 0.20 |
| `--easy-boundary` | flag | Use administrative boundary | False |
| `--buffer` | integer | Buffer distance in meters | adaptive |
| `--density-similarity` | float | Density similarity threshold | dynamic |
| `--cores` | integer | CPU cores for parallel processing | auto-detect |
| `--verbose` | flag | Enable detailed logging | False |

## Methodology

### Smart Buffer Algorithm

The Smart Buffer methodology implements a density-based approach for urban boundary detection:

1. **Feature Extraction**: Download all buildings and impervious surfaces from OpenStreetMap
2. **Centroid Analysis**: Extract unified centroids from urban features
3. **Buffer Creation**: Apply adaptive buffering based on city size characteristics
4. **Density Analysis**: Calculate dynamic density thresholds from actual data distribution
5. **Component Selection**: Select urban components based on density similarity criteria
6. **Grid Optimization**: Generate computational grid only within identified urban extent

### Processing Efficiency

The FRC_URB2D threshold processing provides computational optimization by:
- Fast preliminary urban fraction calculation for each grid cell
- Full parameter computation only for cells exceeding the specified threshold
- Typical efficiency gains of 20-60% in computation time
- Maintains accuracy while reducing processing overhead

## Scientific Background

**Aerodynamic Parameters**: Calculated using established formulations from urban climate research literature, including modified Grimmond & Oke (1999) methodology for roughness length and displacement height calculations.

**Building Height Estimation**: Hierarchical approach using explicit OSM height tags, floor count estimation (3.2m per floor), and building type-specific defaults.

**Quality Control**: Geometric validation includes area conservation for buildings spanning multiple cells, parameter bounds verification, and inter-parameter consistency checks.

## System Requirements

- Python 3.8 or higher
- Memory: 8GB RAM minimum (16GB recommended for high-resolution analysis)
- Storage: 1-10GB depending on city size and resolution
- CPU: Multi-core processor recommended for parallel processing
- Internet connection: Required for OpenStreetMap data download

## Output Files

The software generates the following output files:

- `LCZ_UCP_{city}_{grid}m_smart_buffer.csv`: WRF-compatible CSV format
- `urban_parameters_{city}_{grid}m_smart_buffer.gpkg`: Essential parameters in GeoPackage format
- `detailed_parameters_{city}_{grid}m_smart_buffer.gpkg`: Complete dataset with diagnostic variables
- `analysis_boundary_{city}_smart_buffer.gpkg`: Analysis boundary used for computation

## Performance Characteristics

**Computational Efficiency:**
- Smart Buffer method: 20-80% reduction in grid cells compared to municipal boundaries
- FRC_URB2D threshold processing: 20-60% computation time savings
- Processing speed scales linearly with available CPU cores

**Typical Processing Times:**
- Small cities (< 50 km²): 30 seconds to 2 minutes
- Medium cities (50-200 km²): 2-5 minutes  
- Large metropolitan areas (> 200 km²): 10-20 minutes

## Documentation

Complete methodology and technical documentation available in the `docs/` directory.

## Contributing

Contributions to the project are welcome. Please refer to the contributing guidelines for development procedures and code standards.

## License

This software is distributed under the MIT License. See the LICENSE file for details.

## Citation

If you use UrbanIndex in your research, please cite:

```
UrbanIndex: Urban Morphometric Parameter Calculation for WRF Modeling
Author: [Nicola Manconi]
Version: 1.0.0
URL: https://github.com/NicMan89/Urban-Index-for-WRF
```

## References

Grimmond, C.S.B. and T.R. Oke (1999): Aerodynamic properties of urban areas derived from analysis of surface form. Journal of Applied Meteorology, 38(9), 1262-1292.

Chen, F., et al. (2011): The integrated WRF/urban modelling system: development, evaluation, and applications to urban environmental problems. International Journal of Climatology, 31(2), 273-288.

## Acknowledgments

This software utilizes OpenStreetMap data (© OpenStreetMap contributors) and the OSMnx Python package for network analysis.
