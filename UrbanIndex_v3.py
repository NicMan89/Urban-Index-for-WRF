import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union, nearest_points
from scipy.spatial import cKDTree
import warnings
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
from multiprocessing import Pool, cpu_count
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
warnings.filterwarnings('ignore')

# === GLOBAL FUNCTIONS FOR PARALLEL COMPUTATION ===

def _parallel_cell_worker(args):
    """
    Worker function for parallel computation of cell parameters
    Must be global to be serializable with pickle
    """
    try:
        cell_data, buildings_data, roads_data, impervious_data = args
        
        # Extract FRC threshold from cell data
        frc_threshold = cell_data.get('frc_threshold', 0.20)
        
        # Reconstruct cell
        cell_props = cell_data['properties']
        cell_geometry = cell_data['geometry']
        cell_index = cell_data['index']
        
        # === STEP 1: FAST PRELIMINARY FRC_URB2D CALCULATION ===
        preliminary_frc = _calculate_preliminary_frc_parallel(
            cell_geometry, cell_index, buildings_data, roads_data
        )
        
        # === STEP 2: THRESHOLD-BASED DECISION ===
        if preliminary_frc >= frc_threshold:
            # Reconstruct GeoDataFrames from serialized data
            buildings_with_cells = None
            if buildings_data and buildings_data['data']:
                buildings_with_cells = gpd.GeoDataFrame(buildings_data['data'], crs=buildings_data['crs'])
                
            roads_with_cells = None  
            if roads_data and roads_data['data']:
                roads_with_cells = gpd.GeoDataFrame(roads_data['data'], crs=roads_data['crs'])
                
            impervious_with_cells = None
            if impervious_data and impervious_data['data']:
                impervious_with_cells = gpd.GeoDataFrame(impervious_data['data'], crs=impervious_data['crs'])
            
            # Calculate complete parameters using correct logic
            params = _calculate_cell_parameters_worker_logic(
                cell_geometry, cell_props, cell_index, 
                buildings_with_cells, roads_with_cells, impervious_with_cells
            )
        else:
            # NA parameters for cells below threshold
            params = _na_parameters_static()
        
        # Add cell identifiers and preliminary FRC
        params['cell_id'] = cell_props['cell_id']
        params['x_index'] = cell_props['x_index']  
        params['y_index'] = cell_props['y_index']
        params['preliminary_frc'] = preliminary_frc
        
        return params
        
    except Exception as e:
        # Fallback for errors
        error_info = {
            'cell_id': cell_data['properties'].get('cell_id', 'unknown'),
            'x_index': cell_data['properties'].get('x_index', 0),
            'y_index': cell_data['properties'].get('y_index', 0),
            'error': str(e),
            'error_type': type(e).__name__,
            'preliminary_frc': 0.0,
            **_na_parameters_static()
        }
        return error_info

def _calculate_preliminary_frc_parallel(cell_geometry, cell_index, buildings_data, roads_data):
    """Fast preliminary FRC_URB2D calculation for parallel worker"""
    try:
        cell_bounds = cell_geometry.bounds
        cell_area = cell_geometry.area
        minx, miny, maxx, maxy = cell_bounds
        
        building_area_estimate = 0.0
        road_area_estimate = 0.0
        
        # === BUILDINGS ESTIMATION ===
        if buildings_data and buildings_data['data']:
            for building_dict in buildings_data['data']:
                # Check if building intersects cell (fast bounds filter)
                if ('feature_minx' in building_dict and 'feature_maxx' in building_dict and
                    'feature_miny' in building_dict and 'feature_maxy' in building_dict):
                    
                    bldg_minx = building_dict['feature_minx']
                    bldg_maxx = building_dict['feature_maxx'] 
                    bldg_miny = building_dict['feature_miny']
                    bldg_maxy = building_dict['feature_maxy']
                    
                    # Check overlap
                    if (bldg_minx < maxx and bldg_maxx > minx and 
                        bldg_miny < maxy and bldg_maxy > miny):
                        
                        # Estimate overlap area
                        overlap_x = max(0, min(maxx, bldg_maxx) - max(minx, bldg_minx))
                        overlap_y = max(0, min(maxy, bldg_maxy) - max(miny, bldg_miny))
                        overlap_area = overlap_x * overlap_y
                        
                        building_area = building_dict.get('area_m2', overlap_area)
                        estimated_area = min(overlap_area, building_area)
                        building_area_estimate += estimated_area
        
        # === ROADS ESTIMATION ===
        if roads_data and roads_data['data']:
            for road_dict in roads_data['data']:
                # Fast bounds filter
                if ('feature_minx' in road_dict and 'feature_maxx' in road_dict and
                    'feature_miny' in road_dict and 'feature_maxy' in road_dict):
                    
                    road_minx = road_dict['feature_minx']
                    road_maxx = road_dict['feature_maxx']
                    road_miny = road_dict['feature_miny'] 
                    road_maxy = road_dict['feature_maxy']
                    
                    # Check overlap
                    if (road_minx < maxx and road_maxx > minx and 
                        road_miny < maxy and road_maxy > miny):
                        
                        # Estimate road length in cell
                        road_width = road_dict.get('road_width_m', 5.0)
                        road_length = max(road_maxx - road_minx, road_maxy - road_miny)
                        road_area_estimate += road_length * road_width * 0.5  # Reduction factor
        
        # === PRELIMINARY FRC CALCULATION ===
        total_impervious_estimate = building_area_estimate + road_area_estimate
        preliminary_frc = min(total_impervious_estimate / cell_area, 1.0)
        
        return preliminary_frc
        
    except Exception:
        return 0.0

def _calculate_cell_parameters_worker_logic(cell_geometry, cell_props, cell_index, 
                                           buildings_with_cells, roads_with_cells, impervious_with_cells):
    """
    Calculate parameters for a cell (ALIGNED VERSION for multiprocessing)
    """
    cell_area = cell_geometry.area
    cell_bounds = cell_geometry.bounds
    minx, miny, maxx, maxy = cell_bounds
    
    # === UNIFIED BUILDING FILTERING ===
    cell_buildings = _filter_buildings_for_cell(buildings_with_cells, cell_index, cell_bounds)
    
    if len(cell_buildings) == 0:
        return _na_parameters_static()
    
    # === PRECISE INTERSECTION CALCULATION ===
    building_intersections = []
    total_building_area = 0.0
    
    for idx, building in cell_buildings.iterrows():
        try:
            # Precise geometric intersection
            intersection = building.geometry.intersection(cell_geometry)
            if intersection.is_empty or not intersection.is_valid:
                continue
            
            area = intersection.area
            if area <= 0:
                continue
            
            # Get building parameters with fallback
            height = building.get('height_m', 9.0)
            width = building.get('width_m', 15.0)
            original_area = building.get('area_m2', area)
            
            # Verify consistency
            if area > original_area * 1.01:
                area = min(area, original_area)
            
            # Determine if building is straddling (improved logic)
            crossing_info = _analyze_building_cell_relationship(
                building.geometry.bounds, cell_bounds, area, original_area
            )
            
            building_intersections.append({
                'area': area,
                'height': height,
                'width': width,
                'original_area': original_area,
                'intersection_fraction': area / original_area if original_area > 0 else 1.0,
                'centroid': intersection.centroid,
                'is_straddling': crossing_info['is_straddling'],
                'crossing_type': crossing_info['crossing_type']
            })
            total_building_area += area
            
        except Exception:
            continue
    
    if not building_intersections:
        return _na_parameters_static()
    
    # === PARAMETER CALCULATION ===
    heights = [b['height'] for b in building_intersections]
    areas = [b['area'] for b in building_intersections]
    
    mh_min = min(heights)
    mh_max = max(heights)
    mh_mean = np.average(heights, weights=areas)
    sigma_h = np.sqrt(np.average((np.array(heights) - mh_mean)**2, weights=areas))
    
    bldfr_urb2d = total_building_area / cell_area
    
    widths = [b['width'] for b in building_intersections]
    bw = np.average(widths, weights=areas)
    
    # Estimate street width
    street_width = _analyze_building_distances_simple(cell_bounds, building_intersections, roads_with_cells)
    h2w = mh_mean / max(street_width, 1.0) if street_width > 0 else 0.0
    
    # Calculate FRC_URB2D
    frc_urb2d = _calculate_impervious_fraction_simple(
        cell_bounds, cell_area, total_building_area, cell_buildings, 
        roads_with_cells, impervious_with_cells
    )
    
    # Aerodynamic parameters
    lambda_p = bldfr_urb2d
    z0 = _calculate_roughness_length_static(mh_mean, lambda_p)
    zd = _calculate_displacement_height_static(mh_mean, lambda_p)
    
    return {
        'FRC_URB2D': frc_urb2d,
        'MH_URB2D_MIN': mh_min,
        'MH_URB2D': mh_mean,
        'MH_URB2D_MAX': mh_max,
        'BLDFR_URB2D': bldfr_urb2d,
        'H2W': h2w,
        'BW': bw,
        'Z0': z0,
        'ZD': zd,
        'Sigma': sigma_h,
        'building_count': len(building_intersections),
        'total_building_area': total_building_area,
        'street_width_analyzed': street_width,
        'lambda_p': lambda_p
    }

def _filter_buildings_for_cell(buildings_with_cells, cell_index, cell_bounds):
    """
    UNIFIED building filter for a cell (same for sequential and parallel)
    """
    if buildings_with_cells is None or len(buildings_with_cells) == 0:
        return gpd.GeoDataFrame()
    
    minx, miny, maxx, maxy = cell_bounds
    
    # METHOD 1: Use spatial join results if available
    if 'index_right' in buildings_with_cells.columns:
        mask = buildings_with_cells['index_right'] == cell_index
        return buildings_with_cells[mask].copy()
    
    # METHOD 2: Filtering by pre-calculated coordinates
    elif all(col in buildings_with_cells.columns for col in ['feature_minx', 'feature_maxx', 'feature_miny', 'feature_maxy']):
        mask = (
            (buildings_with_cells['feature_minx'] < maxx) &
            (buildings_with_cells['feature_maxx'] > minx) &
            (buildings_with_cells['feature_miny'] < maxy) &
            (buildings_with_cells['feature_maxy'] > miny)
        )
        return buildings_with_cells[mask].copy()
    
    # METHOD 3: Fallback - calculate bounds on the fly
    else:
        try:
            bounds = buildings_with_cells.bounds
            mask = (
                (bounds['minx'] < maxx) &
                (bounds['maxx'] > minx) &
                (bounds['miny'] < maxy) &
                (bounds['maxy'] > miny)
            )
            return buildings_with_cells[mask].copy()
        except Exception:
            return gpd.GeoDataFrame()

def _analyze_building_cell_relationship(building_bounds, cell_bounds, intersection_area, original_area):
    """
    Analyze geometric relationship between building and cell (IMPROVED LOGIC)
    """
    bldg_minx, bldg_miny, bldg_maxx, bldg_maxy = building_bounds
    cell_minx, cell_miny, cell_maxx, cell_maxy = cell_bounds
    
    # Calculate intersection fraction
    intersection_fraction = intersection_area / original_area if original_area > 0 else 1.0
    
    # === RELATIVE POSITION ANALYSIS ===
    
    # 1. COMPLETE CONTAINMENT
    is_contained = (
        bldg_minx >= cell_minx and bldg_maxx <= cell_maxx and
        bldg_miny >= cell_miny and bldg_maxy <= cell_maxy
    )
    
    # 2. COMPLETE CROSSING (building larger than cell)
    crosses_completely = (
        bldg_minx <= cell_minx and bldg_maxx >= cell_maxx and
        bldg_miny <= cell_miny and bldg_maxy >= cell_maxy
    )
    
    # 3. PARTIAL CROSSING
    crosses_x = (
        (bldg_minx < cell_minx and bldg_maxx > cell_minx) or  # enters from left
        (bldg_minx < cell_maxx and bldg_maxx > cell_maxx)     # exits from right
    )
    
    crosses_y = (
        (bldg_miny < cell_miny and bldg_maxy > cell_miny) or  # enters from bottom
        (bldg_miny < cell_maxy and bldg_maxy > cell_maxy)     # exits from top
    )
    
    # 4. FINAL CLASSIFICATION
    if is_contained:
        crossing_type = 'CONTAINED'
        is_straddling = False
    elif crosses_completely:
        crossing_type = 'FULL_CROSS'
        is_straddling = True
    elif crosses_x and crosses_y:
        crossing_type = 'DIAGONAL_CROSS'
        is_straddling = True
    elif crosses_x:
        crossing_type = 'HORIZONTAL_CROSS'
        is_straddling = True
    elif crosses_y:
        crossing_type = 'VERTICAL_CROSS'
        is_straddling = True
    else:
        # Additional check: if intersection fraction is very low,
        # probably touches only a corner or edge
        if intersection_fraction < 0.1:
            crossing_type = 'EDGE_TOUCH'
            is_straddling = True
        else:
            crossing_type = 'PARTIAL'
            is_straddling = intersection_fraction < 0.95
    
    return {
        'is_straddling': is_straddling,
        'crossing_type': crossing_type,
        'intersection_fraction': intersection_fraction,
        'is_contained': is_contained,
        'crosses_completely': crosses_completely
    }

def _analyze_building_distances_simple(cell_bounds, building_intersections, roads_with_cells):
    """Simplified version for parallel worker"""
    if roads_with_cells is None or len(building_intersections) < 2:
        return 20.0
    
    try:
        minx, miny, maxx, maxy = cell_bounds
        
        if len(roads_with_cells) > 0 and 'road_width_m' in roads_with_cells.columns:
            # Use average road width if available
            mean_width = roads_with_cells['road_width_m'].mean()
            return max(3.0, min(mean_width * 1.5, 50.0))
        else:
            # Fallback: estimate from building density
            cell_area = (maxx - minx) * (maxy - miny)
            building_density = len(building_intersections) / cell_area
            return max(5.0, min(50.0, 1.0 / np.sqrt(building_density))) if building_density > 0 else 20.0
    except:
        return 15.0

def _calculate_impervious_fraction_simple(cell_bounds, cell_area, building_area, 
                                        cell_buildings, roads_with_cells, impervious_with_cells):
    """Simplified version for parallel worker"""
    try:
        # Start with building area
        total_impervious = building_area
        
        minx, miny, maxx, maxy = cell_bounds
        cell_geom = box(minx, miny, maxx, maxy)
        
        # === ROADS ===
        if roads_with_cells is not None and len(roads_with_cells) > 0:
            road_area = 0.0
            
            # Filter roads
            if 'feature_minx' in roads_with_cells.columns:
                mask = (
                    (roads_with_cells['feature_minx'] < maxx) &
                    (roads_with_cells['feature_maxx'] > minx) &
                    (roads_with_cells['feature_miny'] < maxy) &
                    (roads_with_cells['feature_maxy'] > miny)
                )
                cell_roads = roads_with_cells[mask]
            else:
                cell_roads = roads_with_cells
            
            for idx, road in cell_roads.iterrows():
                try:
                    road_width = road.get('road_width_m', 5.0)
                    buffered_road = road.geometry.buffer(road_width / 2.0)
                    road_intersection = buffered_road.intersection(cell_geom)
                    if hasattr(road_intersection, 'area') and road_intersection.area > 0:
                        road_area += road_intersection.area
                except Exception:
                    continue
            
            total_impervious += road_area * 0.9
        
        # === OTHER IMPERVIOUS SURFACES ===
        if impervious_with_cells is not None and len(impervious_with_cells) > 0:
            other_area = 0.0
            
            if 'feature_minx' in impervious_with_cells.columns:
                mask = (
                    (impervious_with_cells['feature_minx'] < maxx) &
                    (impervious_with_cells['feature_maxx'] > minx) &
                    (impervious_with_cells['feature_miny'] < maxy) &
                    (impervious_with_cells['feature_maxy'] > miny)
                )
                cell_impervious = impervious_with_cells[mask]
            else:
                cell_impervious = impervious_with_cells
            
            for idx, surface in cell_impervious.iterrows():
                try:
                    intersection = surface.geometry.intersection(cell_geom)
                    if hasattr(intersection, 'area') and intersection.area > 0:
                        other_area += intersection.area
                except Exception:
                    continue
            
            total_impervious += other_area * 0.7
        
        # Calculate final fraction
        frc_urb2d = min(total_impervious / cell_area, 1.0)
        return frc_urb2d
        
    except Exception:
        return min(building_area / cell_area, 1.0)

def _na_parameters_static():
    """NA parameters (static version for multiprocessing)"""
    return {
        'FRC_URB2D': np.nan,
        'MH_URB2D_MIN': np.nan,
        'MH_URB2D': np.nan,
        'MH_URB2D_MAX': np.nan,
        'BLDFR_URB2D': np.nan,
        'H2W': np.nan,
        'BW': np.nan,
        'Z0': np.nan,
        'ZD': np.nan,
        'Sigma': np.nan,
        'building_count': 0,
        'total_building_area': 0.0,
        'street_width_analyzed': np.nan,
        'lambda_p': np.nan
    }

def _calculate_roughness_length_static(mean_height, lambda_p):
    """Roughness length (static version)"""
    if lambda_p <= 0 or mean_height <= 0:
        return 0.03
    
    if lambda_p < 0.15:
        z0 = mean_height * (0.1 * lambda_p)
    elif lambda_p < 0.35:
        z0 = mean_height * (0.05 + 0.05 * lambda_p / 0.35)
    else:
        z0 = mean_height * 0.1
    
    return max(min(z0, mean_height * 0.2), 0.1)

def _calculate_displacement_height_static(mean_height, lambda_p):
    """Zero-plane displacement height (static version)"""
    if lambda_p <= 0 or mean_height <= 0:
        return 0.0
    
    if lambda_p < 0.15:
        zd = mean_height * (3.0 * lambda_p)
    elif lambda_p < 0.35:
        zd = mean_height * (0.45 + 0.3 * (lambda_p - 0.15) / 0.2)
    else:
        zd = mean_height * 0.75
    
    return max(min(zd, mean_height * 0.8), 0.0)

# === END OF GLOBAL PARALLEL FUNCTIONS ===

class UrbanMorphometricWorkflow:
    """
    SMART BUFFER Workflow to calculate urban morphometric parameters:
    1. Download ALL buildings from city query (no boundary limit)
    2. Download ALL impervious surfaces from city query
    3. Extract unified centroids from buildings + impervious surfaces
    4. Create analysis boundary using SMART BUFFER METHOD with density-based selection
    5. Create grid only within this optimized boundary
    6. Calculate parameters cell by cell with FRC_URB2D threshold
    """
    
    def __init__(self, city_name, grid_size_m=1000, output_dir="output", n_cores=None, debug_mode=False, frc_threshold=0.20):
        self.city_name = city_name
        self.grid_size_m = grid_size_m
        self.output_dir = output_dir
        self.n_cores = n_cores or max(1, cpu_count() - 1)
        self._debug_mode = debug_mode
        self.frc_threshold = frc_threshold
        
        # Downloaded data
        self.buildings = None
        self.roads = None
        self.other_features = {}
        self.unified_centroids = None
        self.analysis_boundary = None  # NEW: smart buffer boundary
        self.grid = None
        
        # Results
        self.results = None
        
        # Options
        self._force_redownload = False
        
        os.makedirs(self.output_dir, exist_ok=True)
        if debug_mode:
            print(f"🏗️ SMART BUFFER Workflow initialized for {city_name}")
            print(f"   Grid: {grid_size_m}m, FRC threshold: ≥{frc_threshold:.0%}")
            print(f"   Parallel computation: {self.n_cores} cores available")
    
    def step1_download_buildings_unrestricted(self):
        """STEP 1: Download ALL buildings from city query (no boundary restrictions)"""
        print(f"\n🏢 STEP 1: Downloading ALL buildings for {self.city_name}")
        
        # Check if buildings file already exists
        city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
        buildings_file = os.path.join(self.output_dir, f"temp_buildings_{city_clean}.gpkg")
        
        if os.path.exists(buildings_file) and not getattr(self, '_force_redownload', False):
            try:
                self.buildings = self._load_temp_file_safe(buildings_file, "buildings")
                if self.buildings is not None:
                    print(f"✅ {len(self.buildings)} buildings loaded from temporary file")
                    return True
                else:
                    print("⚠️ Could not load temporary buildings file, proceeding with download...")
            except Exception:
                print("⚠️ Error loading existing file, proceeding with download...")
        
        try:
            print(f"   🔍 Querying OSM for all buildings in {self.city_name}...")
            
            # Download ALL buildings from city query (no boundary limit)
            buildings_raw = ox.features_from_place(
                self.city_name,
                tags={'building': True}
            )
            
            if buildings_raw.empty:
                print("❌ No buildings found")
                return False
            
            # Filter only valid polygons
            buildings_raw = buildings_raw[buildings_raw.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            buildings_raw = buildings_raw[buildings_raw.geometry.is_valid]
            
            print(f"✅ {len(buildings_raw)} buildings downloaded from OSM")
            
            # Get first building centroid to determine appropriate UTM zone
            first_building = buildings_raw.geometry.iloc[0]
            centroid = first_building.centroid
            lon, lat = centroid.x, centroid.y
            
            # Determine UTM zone automatically
            import math
            utm_zone = int(math.floor((lon + 180) / 6) + 1)
            hemisphere = 'north' if lat >= 0 else 'south'
            
            # Build UTM EPSG
            if hemisphere == 'north':
                utm_epsg = f"EPSG:326{utm_zone:02d}"
            else:
                utm_epsg = f"EPSG:327{utm_zone:02d}"
            
            # Convert to UTM CRS for metric calculations
            self.buildings = buildings_raw.to_crs(utm_epsg)
            
            print(f"   📐 Converted to {utm_epsg} for metric calculations")
            
            # Extract and clean height data
            self._extract_clean_heights()
            
            # Calculate building geometric parameters
            self._calculate_building_geometry()
            
            # Save temporary buildings file with robust error handling
            self._save_buildings_temp_file(buildings_file)
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading buildings: {e}")
            return False
    
    def _extract_clean_heights(self):
        """Extract and clean height data from OSM buildings"""
        heights = []
        height_sources = []
        
        for idx, building in self.buildings.iterrows():
            height = None
            source = "default"
            
            # PRIORITY 1: Explicit height tag
            for height_col in ['height', 'building:height', 'roof:height']:
                if height_col in building and pd.notna(building[height_col]):
                    try:
                        height_str = str(building[height_col]).lower()
                        height_str = height_str.replace('m', '').replace('meters', '').replace('metres', '')
                        height_str = height_str.replace(',', '.').strip()
                        
                        import re
                        numbers = re.findall(r'\d+\.?\d*', height_str)
                        if numbers:
                            height = float(numbers[0])
                            source = height_col
                            break
                    except:
                        continue
            
            # PRIORITY 2: Number of floors
            if height is None:
                for levels_col in ['building:levels', 'levels', 'building:floors', 'floors']:
                    if levels_col in building and pd.notna(building[levels_col]):
                        try:
                            levels = float(building[levels_col])
                            if 1 <= levels <= 50:
                                height = levels * 3.2
                                source = f"{levels_col}_estimated"
                                break
                        except:
                            continue
            
            # PRIORITY 3: Building type
            if height is None:
                building_type = building.get('building', 'yes')
                if isinstance(building_type, list):
                    building_type = building_type[0] if building_type else 'yes'
                
                type_heights = {
                    'house': 6.5, 'detached': 7.0, 'residential': 9.0,
                    'apartments': 12.0, 'commercial': 12.0, 'office': 15.0,
                    'industrial': 8.0, 'school': 12.0, 'hospital': 18.0,
                    'church': 15.0, 'yes': 9.0, 'building': 9.0
                }
                
                height = type_heights.get(building_type, 9.0)
                source = f"type_{building_type}"
            
            heights.append(height)
            height_sources.append(source)
        
        # Add to building data
        self.buildings['height_m'] = heights
        self.buildings['height_source'] = height_sources
        
        if self._debug_mode:
            source_counts = pd.Series(height_sources).value_counts()
            print("📈 Height sources used:")
            for source, count in source_counts.head(5).items():
                percentage = (count / len(heights)) * 100
                print(f"  - {source}: {count} ({percentage:.1f}%)")
    
    def _calculate_building_geometry(self):
        """Calculate geometric parameters of buildings"""
        # Area and perimeter
        self.buildings['area_m2'] = self.buildings.geometry.area
        self.buildings['perimeter_m'] = self.buildings.geometry.length
        
        # Estimated building width
        widths = []
        for idx, building in self.buildings.iterrows():
            try:
                area = building['area_m2']
                perimeter = building['perimeter_m']
                if perimeter > 0:
                    width = (4 * area) / perimeter
                    widths.append(min(width, 200))
                else:
                    widths.append(10.0)
            except:
                widths.append(10.0)
        
        self.buildings['width_m'] = widths
        
        # Compactness
        compactness = []
        for idx, building in self.buildings.iterrows():
            try:
                area = building['area_m2']
                perimeter = building['perimeter_m']
                if perimeter > 0:
                    comp = (4 * np.pi * area) / (perimeter ** 2)
                    compactness.append(comp)
                else:
                    compactness.append(0.0)
            except:
                compactness.append(0.0)
        
        self.buildings['compactness'] = compactness
        
        if self._debug_mode:
            print(f"📊 Building parameters: Average area {self.buildings['area_m2'].mean():.1f} m²")
    
    def step2_download_impervious_surfaces_unrestricted(self):
        """STEP 2: Download ALL impervious surfaces from city query (no boundary restrictions)"""
        print(f"\n🏗️ STEP 2: Downloading ALL impervious surfaces for {self.city_name}")
        
        # Check if impervious surfaces file already exists
        city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
        impervious_file = os.path.join(self.output_dir, f"temp_impervious_{city_clean}.gpkg")
        
        if os.path.exists(impervious_file) and not getattr(self, '_force_redownload', False):
            try:
                impervious_gdf = self._load_temp_file_safe(impervious_file, "impervious surfaces")
                if impervious_gdf is not None:
                    self.other_features['impervious_surfaces'] = impervious_gdf
                    print(f"✅ {len(impervious_gdf)} impervious surfaces loaded from temporary file")
                    return True
                else:
                    print("⚠️ Could not load temporary impervious file, proceeding with download...")
            except Exception:
                print("⚠️ Error loading existing file, proceeding with download...")
        
        try:
            all_impervious = []
            
            # Download different types of impervious surfaces (no boundary limit)
            categories = [
                ({'amenity': ['parking'], 'landuse': ['garages'], 'parking': True}, 'parking'),
                ({'landuse': ['commercial', 'retail', 'industrial']}, 'commercial_industrial'),
                ({'place': ['square'], 'highway': ['pedestrian'], 'amenity': ['marketplace']}, 'squares'),
                ({'aeroway': ['aerodrome', 'runway', 'taxiway'], 'landuse': ['aerodrome']}, 'airport'),
                ({'leisure': ['stadium', 'sports_centre'], 'building': ['stadium']}, 'sports'),
                ({'amenity': ['hospital', 'school'], 'building': ['hospital', 'school']}, 'public'),
                ({'landuse': ['construction'], 'building': ['construction']}, 'construction')
            ]
            
            for tags, surface_type in categories:
                try:
                    print(f"   🔍 Downloading {surface_type} surfaces...")
                    features = ox.features_from_place(self.city_name, tags=tags)
                    if not features.empty:
                        # Filter only polygonal geometries
                        features = features[features.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                        features = features.to_crs(self.buildings.crs)  # Use same CRS as buildings
                        features = features[features.geometry.is_valid]
                        features['surface_type'] = surface_type
                        all_impervious.append(features[['geometry', 'surface_type']])
                        print(f"     ✅ Found {len(features)} {surface_type} features")
                except Exception as e:
                    print(f"     ⚠️ Error downloading {surface_type}: {e}")
                    continue
            
            # Combine all impervious surfaces
            if all_impervious:
                combined_impervious = pd.concat(all_impervious, ignore_index=True)
                combined_impervious = combined_impervious[combined_impervious.geometry.is_valid]
                combined_impervious = combined_impervious.drop_duplicates(subset=['geometry'])
                
                self.other_features['impervious_surfaces'] = combined_impervious
                
                # Save temporary file with robust error handling
                self._save_impervious_temp_file(combined_impervious, impervious_file)
                
                print(f"✅ {len(combined_impervious)} total impervious surfaces downloaded")
                
                if self._debug_mode:
                    type_counts = combined_impervious['surface_type'].value_counts()
                    print(f"   📊 Surfaces by type: {dict(type_counts.head(5))}")
                
                return True
            else:
                print("⚠️ No impervious surfaces found - continuing with buildings only")
                return True  # Not a fatal error
                
        except Exception as e:
            print(f"❌ Error downloading impervious surfaces: {e}")
            print("⚠️ Continuing with buildings only")
            return True  # Not fatal - can work with buildings only
    
    def step3_extract_unified_centroids(self):
        """STEP 3: Extract unified centroids from buildings + impervious surfaces"""
        print(f"\n📍 STEP 3: Extracting unified centroids from all urban features")
        
        if self.buildings is None or len(self.buildings) == 0:
            print("❌ No buildings available for centroid extraction")
            return False
        
        try:
            all_centroids = []
            
            # Extract building centroids
            print(f"   🏢 Extracting centroids from {len(self.buildings)} buildings...")
            building_centroids = self.buildings.geometry.centroid
            building_centroids_gdf = gpd.GeoDataFrame({
                'geometry': building_centroids,
                'feature_type': 'building',
                'source_area': self.buildings.geometry.area
            }, crs=self.buildings.crs)
            all_centroids.append(building_centroids_gdf)
            
            # Extract impervious surface centroids if available
            if 'impervious_surfaces' in self.other_features:
                impervious = self.other_features['impervious_surfaces']
                if len(impervious) > 0:
                    print(f"   🏗️ Extracting centroids from {len(impervious)} impervious surfaces...")
                    impervious_centroids = impervious.geometry.centroid
                    impervious_centroids_gdf = gpd.GeoDataFrame({
                        'geometry': impervious_centroids,
                        'feature_type': 'impervious',
                        'source_area': impervious.geometry.area
                    }, crs=impervious.crs)
                    all_centroids.append(impervious_centroids_gdf)
            
            # Combine all centroids
            self.unified_centroids = pd.concat(all_centroids, ignore_index=True)
            
            # Remove any invalid centroids
            self.unified_centroids = self.unified_centroids[self.unified_centroids.geometry.is_valid]
            
            print(f"✅ Extracted {len(self.unified_centroids)} unified centroids")
            
            # Calculate centroid statistics
            building_count = (self.unified_centroids['feature_type'] == 'building').sum()
            impervious_count = (self.unified_centroids['feature_type'] == 'impervious').sum()
            
            print(f"   📊 Centroid breakdown:")
            print(f"      Buildings: {building_count}")
            print(f"      Impervious surfaces: {impervious_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error extracting centroids: {e}")
            return False
    
    def step4_create_analysis_boundary_smart_buffer(self, buffer_distance_m=None, 
                                                   density_similarity_threshold=None,
                                                   min_area_km2=None, use_easy_boundary=False):
        """
        STEP 4: SMART BUFFER METHOD with adaptive parameters and dynamic density threshold
        OR EASY BOUNDARY method using OSMnx administrative boundary
        
        Args:
            buffer_distance_m: Buffer distance (None = auto-calculate based on city size)
            density_similarity_threshold: Threshold for densities (None = dynamic calculation)
            min_area_km2: Minimum area for a component (None = auto-calculate)
            use_easy_boundary: If True, download city boundary from OSMnx instead of smart buffer
        """
        
        # === EASY BOUNDARY METHOD ===
        if use_easy_boundary:
            return self._create_easy_boundary_from_osmnx()
        
        # === SMART BUFFER METHOD (original code) ===
        
        # === ADAPTIVE PARAMETER CALCULATION ===
        n_centroids = len(self.unified_centroids)
        
        # Auto buffer size based on city size (LARGER for smaller cities)
        if buffer_distance_m is None:
            if n_centroids > 50000:        # Large cities (Rome, Milan)
                buffer_distance_m = 300    # Smaller buffer - avoid over-merging
            elif n_centroids > 20000:      # Medium cities (Florence, Bologna)
                buffer_distance_m = 350    # Medium buffer
            elif n_centroids > 5000:       # Small cities (Bari, Brescia)
                buffer_distance_m = 400    # Larger buffer - more cautious
            else:                          # Very small centers (Aosta, Matera)
                buffer_distance_m = 450    # Maximum buffer - very cautious
        
        # Auto min area based on city size
        if min_area_km2 is None:
            if n_centroids > 30000:
                min_area_km2 = 0.8         # Larger threshold for big cities
            elif n_centroids > 10000:
                min_area_km2 = 0.5         # Medium threshold
            else:
                min_area_km2 = 0.2         # Smaller threshold for small cities
        
        print(f"\n🎯 STEP 4 (Smart Buffer): Creating analysis boundary with adaptive parameters")
        print(f"   City size: {n_centroids} centroids → buffer: {buffer_distance_m}m")
        print(f"   Grid size: {self.grid_size_m}m (for merge distance)")
        print(f"   Min area: {min_area_km2} km²")
        print(f"   Density threshold: DYNAMIC (calculated from actual densities)")
        
        if self.unified_centroids is None or len(self.unified_centroids) == 0:
            print("❌ No centroids available for boundary creation")
            return False
        
        try:
            start_time = time.time()
            
            # === 1. FAST BUFFER + UNION ===
            print(f"   🔵 Creating {buffer_distance_m}m buffers around {len(self.unified_centroids)} centroids...")
            buffered_centroids = self.unified_centroids.geometry.buffer(buffer_distance_m)
            
            print(f"   🔗 Merging overlapping buffers...")
            unified_boundary = unary_union(buffered_centroids.tolist())
            
            buffer_time = time.time() - start_time
            print(f"   ⚡ Buffer + Union time: {buffer_time:.2f}s")
            
            # === 2. ANALYZE COMPONENTS ===
            if hasattr(unified_boundary, 'geoms'):
                components = list(unified_boundary.geoms)
                print(f"   📊 Found {len(components)} separate components")
            else:
                components = [unified_boundary]
                print(f"   📊 Single component found")
            
            if len(components) == 1:
                # Simple case: single component
                self.analysis_boundary = gpd.GeoDataFrame(
                    {'geometry': [components[0]]},
                    crs=self.unified_centroids.crs
                )
                
                area_km2 = components[0].area / 1_000_000
                print(f"   ✅ Single boundary: {area_km2:.1f} km²")
                return True
            
            # === 3. FAST PRE-MERGE USING GRID BUFFER ===
            if len(components) > 10:  # Only for many components (performance optimization)
                print(f"   🚀 Fast pre-merge: buffering {len(components)} components by {self.grid_size_m}m...")
                pre_merge_start = time.time()
                
                # Buffer each component by grid size and merge intersecting ones
                buffered_components = [comp.buffer(self.grid_size_m) for comp in components]
                pre_merged_boundary = unary_union(buffered_components)
                
                # Extract final components after pre-merge
                if hasattr(pre_merged_boundary, 'geoms'):
                    components = list(pre_merged_boundary.geoms)
                    print(f"      ✅ Pre-merge: {len(components)} components after grid-buffer union")
                else:
                    components = [pre_merged_boundary]
                    print(f"      ✅ Pre-merge: Single component after grid-buffer union")
                
                pre_merge_time = time.time() - pre_merge_start
                print(f"      ⚡ Pre-merge time: {pre_merge_time:.2f}s")
                
                # If single component after pre-merge, we're done
                if len(components) == 1:
                    # Erode back to remove artificial buffer
                    final_boundary = components[0].buffer(-self.grid_size_m * 0.8)  # Slight erosion
                    
                    self.analysis_boundary = gpd.GeoDataFrame(
                        {'geometry': [final_boundary]},
                        crs=self.unified_centroids.crs
                    )
                    
                    area_km2 = final_boundary.area / 1_000_000
                    centroids_in_boundary = self.unified_centroids[
                        self.unified_centroids.geometry.within(final_boundary)
                    ]
                    capture_rate = len(centroids_in_boundary) / len(self.unified_centroids) * 100
                    
                    total_time = time.time() - start_time
                    print(f"   🎉 Single boundary after pre-merge - ANALYSIS COMPLETE:")
                    print(f"      Total time: {total_time:.2f}s")
                    print(f"      Final area: {area_km2:.1f} km²")
                    print(f"      Centroids captured: {len(centroids_in_boundary)}/{len(self.unified_centroids)} ({capture_rate:.1f}%)")
                    print(f"      ⚡ DENSITY ANALYSIS SKIPPED (single zone remaining)")
                    
                    return True
            
            # === 4. ADVANCED COMPONENT ANALYSIS (only for remaining components) ===
            component_info = self._analyze_boundary_components(components, min_area_km2)
            
            if not component_info:
                print("❌ No valid components found")
                return False
            
            # === 5. DISTANCE-BASED MERGING ===
            merged_components = self._merge_close_components(component_info, self.grid_size_m)
            
            # === 6. DYNAMIC DENSITY THRESHOLD CALCULATION ===
            component_densities = [comp['density'] for comp in merged_components]
            dynamic_similarity_threshold = self._calculate_dynamic_density_threshold(component_densities)
            
            if density_similarity_threshold is None:
                density_similarity_threshold = dynamic_similarity_threshold
            
            print(f"   🧮 Dynamic density threshold: ±{density_similarity_threshold*100:.1f}%")
            
            # === 7. DENSITY-BASED SELECTION ===
            final_boundary = self._select_components_by_density(
                merged_components, density_similarity_threshold
            )
            
            if final_boundary is None:
                print("❌ Could not create final boundary")
                return False
            
            # === 8. CREATE FINAL BOUNDARY ===
            self.analysis_boundary = gpd.GeoDataFrame(
                {'geometry': [final_boundary]},
                crs=self.unified_centroids.crs
            )
            
            # === 9. STATISTICS ===
            total_time = time.time() - start_time
            boundary_area_km2 = final_boundary.area / 1_000_000
            centroids_in_boundary = self.unified_centroids[
                self.unified_centroids.geometry.within(final_boundary)
            ]
            capture_rate = len(centroids_in_boundary) / len(self.unified_centroids) * 100
            final_density = len(centroids_in_boundary) / boundary_area_km2
            
            print(f"   🎉 Smart buffer boundary created:")
            print(f"      Total time: {total_time:.2f}s")
            print(f"      Buffer used: {buffer_distance_m}m (adaptive)")
            print(f"      Density threshold: ±{density_similarity_threshold*100:.1f}% (dynamic)")
            print(f"      Final area: {boundary_area_km2:.1f} km²")
            print(f"      Centroids captured: {len(centroids_in_boundary)}/{len(self.unified_centroids)} ({capture_rate:.1f}%)")
            print(f"      Final density: {final_density:.1f} centroids/km²")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating smart buffer boundary: {e}")
            if self._debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def _calculate_dynamic_density_threshold(self, component_densities):
        """
        Calculate dynamic density similarity threshold based on actual density range
        """
        if len(component_densities) < 2:
            return 0.20  # Fallback for single component
        
        min_density = min(component_densities)
        max_density = max(component_densities)
        density_range = max_density - min_density
        
        print(f"      Density analysis: min={min_density:.1f}, max={max_density:.1f}, range={density_range:.1f}")
        
        if density_range == 0:
            # All densities identical
            print(f"      All densities identical → threshold: 10%")
            return 0.10
        else:
            # Dynamic threshold based on range
            relative_range = density_range / max_density
            
            # Adaptive threshold with reasonable bounds
            if relative_range < 0.20:      # Very similar densities
                threshold = 0.15
                reason = "similar densities"
            elif relative_range < 0.50:    # Moderate differences  
                threshold = 0.25
                reason = "moderate differences"
            else:                          # Large differences
                threshold = 0.35
                reason = "large differences"
            
            print(f"      Relative range: {relative_range:.1f} ({reason}) → threshold: {threshold*100:.0f}%")
            return threshold
    
    def _analyze_boundary_components(self, components, min_area_km2):
        """Analyze each boundary component for area, centroids, and density (OPTIMIZED VERSION)"""
        component_info = []
        
        print(f"      🔍 Analyzing {len(components)} components (optimized method)...")
        
        # Pre-calculate centroids spatial index for faster queries
        if len(self.unified_centroids) > 10000:  # Only for large datasets
            print(f"      📊 Building spatial index for {len(self.unified_centroids)} centroids...")
            from scipy.spatial import cKDTree
            coords = np.array([[p.x, p.y] for p in self.unified_centroids.geometry])
            spatial_tree = cKDTree(coords)
        else:
            spatial_tree = None
        
        for i, comp in enumerate(components):
            area_km2 = comp.area / 1_000_000
            
            # Skip components too small
            if area_km2 < min_area_km2:
                print(f"      Component {i+1}: {area_km2:.2f} km² (skipped - too small)")
                continue
            
            # === OPTIMIZED POINT COUNTING ===
            if spatial_tree is not None and len(self.unified_centroids) > 50000:
                # FAST METHOD: Use bounding box + spatial tree for large datasets
                minx, miny, maxx, maxy = comp.bounds
                
                # Get candidates within bounding box using spatial tree
                bbox_mask = (
                    (coords[:, 0] >= minx) & (coords[:, 0] <= maxx) &
                    (coords[:, 1] >= miny) & (coords[:, 1] <= maxy)
                )
                candidate_indices = np.where(bbox_mask)[0]
                
                if len(candidate_indices) > 0:
                    # Sample for very large candidate sets to speed up
                    if len(candidate_indices) > 10000:
                        sample_size = min(5000, len(candidate_indices))
                        candidate_indices = np.random.choice(candidate_indices, sample_size, replace=False)
                        print(f"      Component {i+1}: Sampling {sample_size} of {len(np.where(bbox_mask)[0])} candidates")
                    
                    # Check actual containment only for candidates
                    candidate_points = self.unified_centroids.iloc[candidate_indices]
                    points_inside = candidate_points[candidate_points.geometry.within(comp)]
                    
                    # Estimate total count if sampled
                    if len(candidate_indices) < len(np.where(bbox_mask)[0]):
                        scale_factor = len(np.where(bbox_mask)[0]) / len(candidate_indices)
                        estimated_count = int(len(points_inside) * scale_factor)
                        point_count = estimated_count
                        print(f"      Component {i+1}: Estimated {point_count} points (from {len(points_inside)} sampled)")
                    else:
                        point_count = len(points_inside)
                        print(f"      Component {i+1}: Exact {point_count} points")
                else:
                    points_inside = gpd.GeoDataFrame()
                    point_count = 0
                    
            else:
                # STANDARD METHOD: Direct spatial operation for smaller datasets
                points_inside = self.unified_centroids[self.unified_centroids.geometry.within(comp)]
                point_count = len(points_inside)
            
            # Calculate density metrics
            density = point_count / area_km2 if area_km2 > 0 else 0
            
            # Weighted density (use sample for estimation if needed)
            if len(points_inside) > 0:
                sample_size = min(1000, len(points_inside))  # Sample for speed
                points_sample = points_inside.sample(n=sample_size) if len(points_inside) > sample_size else points_inside
                avg_source_area = points_sample['source_area'].mean()
                weighted_point_count = (avg_source_area / 1000) * point_count  # Estimate total
                weighted_density = weighted_point_count / area_km2 if area_km2 > 0 else 0
            else:
                weighted_density = 0
            
            component_info.append({
                'index': i,
                'geometry': comp,
                'area_km2': area_km2,
                'point_count': point_count,
                'density': density,
                'weighted_density': weighted_density,
                'points_inside': points_inside  # Keep for final selection
            })
            
            print(f"      Component {i+1}: {area_km2:.1f} km², {point_count} points, {density:.1f} pts/km²")
        
        return component_info
    
    def _merge_close_components(self, component_info, merge_distance_m):
        """Merge components that are closer than grid size (OPTIMIZED VERSION)"""
        print(f"   🔗 Checking distances between {len(component_info)} components (merge if ≤ {merge_distance_m}m)...")
        
        if len(component_info) <= 1:
            return component_info
        
        # === FAST MERGE USING BUFFERING ===
        if len(component_info) > 5:  # Use fast method for many components
            print(f"      🚀 Fast merge method: buffering components...")
            
            # Buffer each component and find intersections
            buffered_components = []
            for comp_info in component_info:
                buffered_geom = comp_info['geometry'].buffer(merge_distance_m / 2)
                buffered_components.append({
                    **comp_info,
                    'buffered_geometry': buffered_geom
                })
            
            # Find groups based on buffer intersections
            groups = []
            used = set()
            
            for i, comp_a in enumerate(buffered_components):
                if i in used:
                    continue
                
                # Start new group
                group = [i]
                used.add(i)
                
                # Find all components that intersect with this one
                for j, comp_b in enumerate(buffered_components):
                    if j not in used and comp_a['buffered_geometry'].intersects(comp_b['buffered_geometry']):
                        group.append(j)
                        used.add(j)
                        print(f"      Merging components {comp_a['index']+1} and {comp_b['index']+1} (buffered intersection)")
                
                groups.append(group)
            
        else:
            # === STANDARD METHOD FOR FEW COMPONENTS ===
            # Calculate distance matrix
            n = len(component_info)
            distance_matrix = np.full((n, n), np.inf)
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = component_info[i]['geometry'].distance(component_info[j]['geometry'])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            # Find groups to merge
            groups = []
            used = set()
            
            for i in range(n):
                if i in used:
                    continue
                
                # Start new group
                group = [i]
                used.add(i)
                
                # Find components within merge distance
                for j in range(n):
                    if j not in used and distance_matrix[i, j] <= merge_distance_m:
                        group.append(j)
                        used.add(j)
                        print(f"      Merging components {i+1} and {j+1} (distance: {distance_matrix[i, j]:.0f}m)")
                
                groups.append(group)
        
        # === CREATE MERGED COMPONENTS ===
        merged_components = []
        for group_indices in groups:
            if len(group_indices) == 1:
                # Single component
                merged_components.append(component_info[group_indices[0]])
            else:
                # Merge multiple components
                group_geoms = [component_info[i]['geometry'] for i in group_indices]
                merged_geom = unary_union(group_geoms)
                
                # Combine statistics (optimized)
                total_area = sum(component_info[i]['area_km2'] for i in group_indices)
                total_points = sum(component_info[i]['point_count'] for i in group_indices)
                total_weighted = sum(component_info[i]['weighted_density'] * component_info[i]['area_km2'] for i in group_indices)
                
                # Recalculate actual area after merge
                actual_area_km2 = merged_geom.area / 1_000_000
                actual_density = total_points / actual_area_km2 if actual_area_km2 > 0 else 0
                actual_weighted_density = total_weighted / actual_area_km2 if actual_area_km2 > 0 else 0
                
                # Combine points_inside (sample if too large)
                all_points_list = [component_info[i]['points_inside'] for i in group_indices if len(component_info[i]['points_inside']) > 0]
                if all_points_list:
                    if sum(len(pts) for pts in all_points_list) > 10000:  # Sample for performance
                        # Take sample from each component proportionally
                        sampled_points = []
                        for pts in all_points_list:
                            sample_size = min(1000, len(pts))
                            sampled_points.append(pts.sample(n=sample_size) if len(pts) > sample_size else pts)
                        all_points = pd.concat(sampled_points)
                    else:
                        all_points = pd.concat(all_points_list)
                else:
                    all_points = gpd.GeoDataFrame()
                
                merged_components.append({
                    'index': f"merged_{'-'.join(str(component_info[i]['index']) for i in group_indices)}",
                    'geometry': merged_geom,
                    'area_km2': actual_area_km2,
                    'point_count': total_points,
                    'density': actual_density,
                    'weighted_density': actual_weighted_density,
                    'points_inside': all_points
                })
                
                print(f"      Merged group: {actual_area_km2:.1f} km², {total_points} points, {actual_density:.1f} pts/km²")
        
        print(f"   ✅ After merging: {len(merged_components)} component(s)")
        return merged_components
    
    def _select_components_by_density(self, components, density_similarity_threshold):
        """Select components based on density similarity and weighted importance"""
        if len(components) == 1:
            return components[0]['geometry']
        
        print(f"   🎯 Selecting components by density (similarity threshold: ±{density_similarity_threshold*100:.0f}%)")
        
        # Calculate weighted density score for each component
        # Score = density * log(point_count) to favor both dense and large components
        for comp in components:
            comp['density_score'] = comp['density'] * np.log(max(comp['point_count'], 1))
            print(f"      Component {comp['index']}: density={comp['density']:.1f}, "
                  f"points={comp['point_count']}, score={comp['density_score']:.2f}")
        
        # Sort by density score
        components_sorted = sorted(components, key=lambda x: x['density_score'], reverse=True)
        best_component = components_sorted[0]
        
        # Find components with similar density (within threshold)
        similar_components = [best_component]
        best_density = best_component['density']
        
        for comp in components_sorted[1:]:
            density_diff = abs(comp['density'] - best_density) / best_density
            if density_diff <= density_similarity_threshold:
                similar_components.append(comp)
                print(f"      Component {comp['index']} has similar density (diff: {density_diff*100:.1f}%)")
            else:
                print(f"      Component {comp['index']} density too different (diff: {density_diff*100:.1f}%)")
        
        if len(similar_components) == 1:
            print(f"   🎯 Selected single best component: {best_component['area_km2']:.1f} km²")
            return best_component['geometry']
        else:
            # Multiple components with similar density - union them
            similar_geoms = [comp['geometry'] for comp in similar_components]
            unified_boundary = unary_union(similar_geoms)
            
            total_area = sum(comp['area_km2'] for comp in similar_components)
            total_points = sum(comp['point_count'] for comp in similar_components)
            
            print(f"   🔗 Unified {len(similar_components)} similar components: "
                  f"{total_area:.1f} km², {total_points} points")
            
            return unified_boundary
    
    def _create_easy_boundary_from_osmnx(self):
        """
        EASY BOUNDARY METHOD: Download city administrative boundary from OSMnx
        """
        print(f"\n🏛️ STEP 4 (Easy Boundary): Downloading administrative boundary from OSMnx")
        print(f"   City: {self.city_name}")
        print(f"   Method: OSMnx geocoding + administrative boundary")
        
        try:
            start_time = time.time()
            
            # === 1. DOWNLOAD CITY BOUNDARY FROM OSMNX ===
            print(f"   🔍 Querying OSMnx for administrative boundary...")
            
            # Try to get the city boundary using geocoding
            try:
                city_boundary_gdf = ox.geocode_to_gdf(self.city_name)
                print(f"   ✅ Found boundary using geocoding")
            except Exception as e1:
                print(f"   ⚠️ Geocoding failed: {e1}")
                print(f"   🔍 Trying alternative method with administrative features...")
                
                # Fallback: try to get administrative boundary features
                try:
                    admin_features = ox.features_from_place(
                        self.city_name, 
                        tags={
                            'admin_level': ['4', '6', '8', '9', '10'],  # Different administrative levels
                            'boundary': 'administrative'
                        }
                    )
                    
                    if not admin_features.empty:
                        # Filter only polygon geometries
                        admin_features = admin_features[admin_features.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                        
                        if not admin_features.empty:
                            # Take the largest polygon (most likely the city boundary)
                            admin_features['area'] = admin_features.geometry.area
                            city_boundary_gdf = admin_features.loc[admin_features['area'].idxmax():admin_features['area'].idxmax()]
                            print(f"   ✅ Found boundary using administrative features")
                        else:
                            raise Exception("No polygon administrative features found")
                    else:
                        raise Exception("No administrative features found")
                        
                except Exception as e2:
                    print(f"   ❌ Administrative features failed: {e2}")
                    print(f"   🔍 Trying place features as last resort...")
                    
                    # Last resort: try place features
                    try:
                        place_features = ox.features_from_place(
                            self.city_name,
                            tags={'place': ['city', 'town', 'municipality']}
                        )
                        
                        if not place_features.empty:
                            place_features = place_features[place_features.geometry.type.isin(['Polygon', 'MultiPolygon'])]
                            if not place_features.empty:
                                place_features['area'] = place_features.geometry.area
                                city_boundary_gdf = place_features.loc[place_features['area'].idxmax():place_features['area'].idxmax()]
                                print(f"   ✅ Found boundary using place features")
                            else:
                                raise Exception("No polygon place features found")
                        else:
                            raise Exception("No place features found")
                            
                    except Exception as e3:
                        print(f"   ❌ All methods failed. Error: {e3}")
                        return False
            
            # === 2. CONVERT TO SAME CRS AS BUILDINGS ===
            if self.buildings is not None:
                target_crs = self.buildings.crs
            else:
                # Default to UTM zone for the city
                first_geom = city_boundary_gdf.geometry.iloc[0]
                centroid = first_geom.centroid
                lon, lat = centroid.x, centroid.y
                
                # Determine UTM zone
                import math
                utm_zone = int(math.floor((lon + 180) / 6) + 1)
                hemisphere = 'north' if lat >= 0 else 'south'
                
                if hemisphere == 'north':
                    target_crs = f"EPSG:326{utm_zone:02d}"
                else:
                    target_crs = f"EPSG:327{utm_zone:02d}"
            
            # Convert to target CRS
            city_boundary_gdf = city_boundary_gdf.to_crs(target_crs)
            
            # === 3. CREATE ANALYSIS BOUNDARY ===
            self.analysis_boundary = city_boundary_gdf[['geometry']].copy()
            
            # === 4. CALCULATE STATISTICS ===
            boundary_geom = self.analysis_boundary.geometry.iloc[0]
            boundary_area_km2 = boundary_geom.area / 1_000_000
            
            # Count centroids within boundary
            if self.unified_centroids is not None:
                centroids_in_boundary = self.unified_centroids[
                    self.unified_centroids.geometry.within(boundary_geom)
                ]
                capture_rate = len(centroids_in_boundary) / len(self.unified_centroids) * 100
                density = len(centroids_in_boundary) / boundary_area_km2
            else:
                centroids_in_boundary = []
                capture_rate = 0
                density = 0
            
            total_time = time.time() - start_time
            
            print(f"   🎉 Easy boundary created successfully:")
            print(f"      Total time: {total_time:.2f}s")
            print(f"      Method: OSMnx administrative boundary")
            print(f"      Boundary area: {boundary_area_km2:.1f} km²")
            print(f"      Centroids captured: {len(centroids_in_boundary)}/{len(self.unified_centroids) if self.unified_centroids is not None else 0} ({capture_rate:.1f}%)")
            print(f"      Centroid density: {density:.1f} centroids/km²")
            print(f"      💡 Using official administrative boundary (faster but may include rural areas)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating easy boundary: {e}")
            if self._debug_mode:
                import traceback
                traceback.print_exc()
            print(f"💡 Suggestion: Try using smart buffer method instead (remove --easy-boundary)")
            return False
    
    def step5_create_optimized_grid(self):
        """STEP 5: Create regular grid ONLY within the analysis boundary"""
        print(f"\n🔲 STEP 5: Creating optimized grid {self.grid_size_m}m within analysis boundary")
        
        if self.analysis_boundary is None:
            print("❌ Analysis boundary not available")
            return False
        
        try:
            # Get bounds of analysis boundary
            bounds = self.analysis_boundary.total_bounds
            minx, miny, maxx, maxy = bounds
            
            # Slightly expand bounds for complete coverage
            buffer_m = self.grid_size_m * 0.5
            minx -= buffer_m
            miny -= buffer_m
            maxx += buffer_m
            maxy += buffer_m
            
            # Generate grid coordinates
            x_coords = np.arange(minx, maxx, self.grid_size_m)
            y_coords = np.arange(miny, maxy, self.grid_size_m)
            
            print(f"   📐 Potential grid: {len(x_coords)} x {len(y_coords)} cells")
            
            # Create cells
            grid_cells = []
            analysis_boundary_geom = self.analysis_boundary.geometry.iloc[0]
            
            for i, x in enumerate(x_coords[:-1]):
                for j, y in enumerate(y_coords[:-1]):
                    cell_geom = box(x, y, x + self.grid_size_m, y + self.grid_size_m)
                    
                    # Check if cell intersects analysis boundary
                    if cell_geom.intersects(analysis_boundary_geom):
                        intersection = cell_geom.intersection(analysis_boundary_geom)
                        
                        if hasattr(intersection, 'area') and intersection.area > 0:
                            intersection_fraction = intersection.area / (self.grid_size_m ** 2)
                            
                            # Keep cells with at least 10% intersection
                            if intersection_fraction >= 0.1:
                                grid_cells.append({
                                    'geometry': cell_geom,
                                    'cell_id': f"{i:04d}_{j:04d}",
                                    'x_index': i,
                                    'y_index': j,
                                    'center_x': x + self.grid_size_m/2,
                                    'center_y': y + self.grid_size_m/2,
                                    'intersection_area': intersection.area,
                                    'intersection_fraction': intersection_fraction
                                })
            
            # Convert to GeoDataFrame
            self.grid = gpd.GeoDataFrame(grid_cells, crs=self.analysis_boundary.crs)
            
            grid_area_km2 = self.grid.geometry.area.sum() / 1_000_000
            boundary_area_km2 = self.analysis_boundary.geometry.area.iloc[0] / 1_000_000
            coverage = (grid_area_km2 / boundary_area_km2) * 100
            
            print(f"✅ Optimized grid created:")
            print(f"   {len(self.grid)} active cells")
            print(f"   Grid coverage: {coverage:.1f}% of analysis boundary")
            print(f"   Total grid area: {grid_area_km2:.1f} km²")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating optimized grid: {e}")
            return False

    def step6_download_roads(self):
        """STEP 6: Download road network for analysis"""
        print(f"\n🛣️ STEP 6: Downloading road network for {self.city_name}")
        
        # Check if roads file already exists
        city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
        roads_file = os.path.join(self.output_dir, f"temp_roads_{city_clean}.gpkg")
        
        if os.path.exists(roads_file) and not getattr(self, '_force_redownload', False):
            try:
                self.roads = self._load_temp_file_safe(roads_file, "roads")
                if self.roads is not None:
                    # Convert to same CRS as buildings/grid
                    self.roads = self.roads.to_crs(self.grid.crs)
                    print(f"✅ {len(self.roads)} road segments loaded from temporary file")
                    return True
                else:
                    print("⚠️ Could not load temporary roads file, proceeding with download...")
            except Exception:
                print("⚠️ Error loading existing file, proceeding with download...")
        
        try:
            # Download road graph
            G = ox.graph_from_place(self.city_name, network_type='all')
            
            # Convert to GeoDataFrame of edges
            nodes, edges = ox.graph_to_gdfs(G)
            
            # Convert to same CRS as grid
            self.roads = edges.to_crs(self.grid.crs)
            
            # Filter out very short roads
            self.roads = self.roads[self.roads.geometry.length > 5.0]
            
            # Classify roads by importance
            self._classify_roads()
            
            print(f"✅ {len(self.roads)} road segments downloaded")
            
            # Save temporary roads file
            self._save_roads_temp_file(roads_file)
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading roads: {e}")
            return False
    
    def _classify_roads(self):
        """Classify roads by type and importance"""
        
        # Road type mapping
        road_hierarchy = {
            'motorway': 1, 'trunk': 2, 'primary': 3, 'secondary': 4,
            'tertiary': 5, 'residential': 6, 'unclassified': 7,
            'service': 8, 'footway': 9, 'cycleway': 9, 'path': 10
        }
        
        # Typical road widths
        road_widths = {
            'motorway': 15, 'trunk': 12, 'primary': 10, 'secondary': 8,
            'tertiary': 7, 'residential': 6, 'unclassified': 5,
            'service': 4, 'footway': 2, 'cycleway': 2, 'path': 1.5
        }
        
        hierarchies = []
        widths = []
        cleaned_highway_types = []
        
        for idx, road in self.roads.iterrows():
            highway_type = road.get('highway', 'unclassified')
            
            # Handle lists in highway column
            if isinstance(highway_type, list):
                highway_type = highway_type[0] if highway_type else 'unclassified'
            
            # Ensure it's a string
            if not isinstance(highway_type, str):
                highway_type = str(highway_type) if highway_type is not None else 'unclassified'
            
            hierarchy = road_hierarchy.get(highway_type, 7)
            width = road_widths.get(highway_type, 5)
            
            hierarchies.append(hierarchy)
            widths.append(width)
            cleaned_highway_types.append(highway_type)
        
        self.roads['road_hierarchy'] = hierarchies
        self.roads['road_width_m'] = widths
        self.roads['highway'] = cleaned_highway_types
    
    def step6_5_process_unified_features(self):
        """STEP 6.5: Process and unify all urban features to avoid overlaps and double counting"""
        print(f"\n🔗 STEP 6.5: Processing unified urban features")
        print(f"   Purpose: Resolve overlaps between buildings, roads, and impervious surfaces")
        
        if self.buildings is None or len(self.buildings) == 0:
            print("❌ No buildings available for feature unification")
            return False
        
        try:
            start_time = time.time()
            
            # === 1. PREPARE ALL FEATURE TYPES ===
            print(f"   📋 Preparing feature layers for unification...")
            
            # Buildings (already processed)
            buildings_count = len(self.buildings)
            print(f"      🏢 Buildings: {buildings_count}")
            
            # Roads (buffered by width)
            roads_count = 0
            if self.roads is not None and len(self.roads) > 0:
                roads_count = len(self.roads)
                print(f"      🛣️ Roads: {roads_count}")
            
            # Other impervious surfaces
            impervious_count = 0
            if 'impervious_surfaces' in self.other_features:
                impervious = self.other_features['impervious_surfaces']
                if len(impervious) > 0:
                    impervious_count = len(impervious)
                    print(f"      🏗️ Impervious surfaces: {impervious_count}")
            
            # === 2. FEATURE PRIORITY HIERARCHY ===
            print(f"   🔄 Applying feature priority hierarchy:")
            print(f"      Priority 1: Buildings (highest accuracy)")
            print(f"      Priority 2: Roads (buffered by width)")  
            print(f"      Priority 3: Other impervious surfaces")
            
            # === 3. CREATE UNIFIED FEATURE DATASET ===
            print(f"   🗂️ Creating unified feature dataset...")
            
            all_features = []
            
            # Add buildings with priority 1
            for idx, building in self.buildings.iterrows():
                all_features.append({
                    'geometry': building.geometry,
                    'feature_type': 'building',
                    'priority': 1,
                    'area_m2': building.get('area_m2', building.geometry.area),
                    'height_m': building.get('height_m', 9.0),
                    'source': 'OSM_buildings'
                })
            
            # Add roads with priority 2 (buffered by width)
            if self.roads is not None and len(self.roads) > 0:
                for idx, road in self.roads.iterrows():
                    road_width = road.get('road_width_m', 5.0)
                    buffered_road = road.geometry.buffer(road_width / 2.0)
                    
                    all_features.append({
                        'geometry': buffered_road,
                        'feature_type': 'road',
                        'priority': 2,
                        'area_m2': buffered_road.area,
                        'road_width_m': road_width,
                        'highway': road.get('highway', 'unclassified'),
                        'source': 'OSM_roads'
                    })
            
            # Add other impervious surfaces with priority 3
            if 'impervious_surfaces' in self.other_features:
                impervious = self.other_features['impervious_surfaces']
                if len(impervious) > 0:
                    for idx, surface in impervious.iterrows():
                        all_features.append({
                            'geometry': surface.geometry,
                            'feature_type': 'impervious',
                            'priority': 3,
                            'area_m2': surface.geometry.area,
                            'surface_type': surface.get('surface_type', 'unknown'),
                            'source': 'OSM_impervious'
                        })
            
            # Convert to GeoDataFrame
            self.unified_features = gpd.GeoDataFrame(all_features, crs=self.buildings.crs)
            
            # === 4. SPATIAL PREPROCESSING FOR CELL-BASED PROCESSING ===
            print(f"   🗂️ Adding spatial optimization columns...")
            
            # Add bounding box coordinates for fast filtering
            bounds = self.unified_features.bounds
            self.unified_features['feature_minx'] = bounds['minx']
            self.unified_features['feature_miny'] = bounds['miny']
            self.unified_features['feature_maxx'] = bounds['maxx']
            self.unified_features['feature_maxy'] = bounds['maxy']
            
            # Add feature area for quick calculations
            self.unified_features['feature_area'] = self.unified_features.geometry.area
            
            processing_time = time.time() - start_time
            total_features = len(self.unified_features)
            
            print(f"   ✅ Unified features processing completed:")
            print(f"      Total unified features: {total_features}")
            print(f"      Buildings: {buildings_count} ({buildings_count/total_features*100:.1f}%)")
            print(f"      Roads: {roads_count} ({roads_count/total_features*100:.1f}%)")
            print(f"      Impervious: {impervious_count} ({impervious_count/total_features*100:.1f}%)")
            print(f"      Processing time: {processing_time:.2f}s")
            print(f"      💡 Overlap resolution will be handled per-cell to avoid memory issues")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing unified features: {e}")
            if self._debug_mode:
                import traceback
                traceback.print_exc()
            return False
    
    def step7_calculate_parameters_per_cell(self, use_parallel=True, debug_parallel=False):
        """
        STEP 7: Calculate all morphometric parameters cell by cell with FRC_URB2D threshold
        """
        print(f"\n🧮 STEP 7: Calculating morphometric parameters for {len(self.grid)} cells")
        print(f"  🎯 FRC_URB2D threshold: ≥{self.frc_threshold:.0%} (cells below threshold will get NA values)")
        
        if any(data is None for data in [self.grid, self.buildings]):
            print("❌ Required data not available")
            return False
        
        start_time = time.time()
        
        # === DEBUG MODE: Compare sequential vs parallel ===
        if debug_parallel:
            print("🔍 DEBUG MODE: Comparing sequential vs parallel with threshold...")
            
            # Test on small subset for debugging
            test_grid = self.grid.head(3).copy()
            original_grid = self.grid
            self.grid = test_grid
            
            print("\n1️⃣ Testing sequential calculation...")
            seq_results = self._calculate_parameters_sequential()
            
            print("\n2️⃣ Testing parallel calculation...")  
            par_results = self._calculate_parameters_parallel()
            
            # Compare results
            self._compare_results(seq_results, par_results)
            
            # Restore original grid
            self.grid = original_grid
            use_parallel = True
        
        # === FINAL CALCULATION ===
        if use_parallel and self.n_cores > 1 and len(self.grid) > 10:
            print(f"🚀 Parallel calculation with threshold {self.frc_threshold:.0%} on {self.n_cores} cores...")
            results = self._calculate_parameters_parallel()
        else:
            print(f"🔄 Sequential calculation with threshold {self.frc_threshold:.0%}...")
            results = self._calculate_parameters_sequential()
        
        if not results:
            print("❌ No results obtained")
            return False
        
        # Analyze threshold effectiveness
        self._analyze_threshold_effectiveness(results)
        
        # Create results GeoDataFrame
        results_df = pd.DataFrame(results)
        self.results = gpd.GeoDataFrame(
            results_df,
            geometry=self.grid.geometry,
            crs=self.grid.crs
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Parameter calculation completed in {elapsed:.1f}s ({len(self.grid)/elapsed:.1f} cells/sec)")
        
        return True
    
    def _analyze_threshold_effectiveness(self, results):
        """Analyze effectiveness of applied threshold"""
        results_df = pd.DataFrame(results)
        
        total_cells = len(results_df)
        calculated_cells = results_df['FRC_URB2D'].notna().sum()
        skipped_cells = total_cells - calculated_cells
        
        efficiency_pct = (skipped_cells / total_cells) * 100
        
        print(f"\n📊 THRESHOLD {self.frc_threshold:.0%} EFFECTIVENESS ANALYSIS:")
        print(f"  Total cells: {total_cells}")
        print(f"  Calculated cells: {calculated_cells} ({(calculated_cells/total_cells)*100:.1f}%)")
        print(f"  Skipped cells: {skipped_cells} ({efficiency_pct:.1f}%)")
        print(f"  Computational efficiency: +{efficiency_pct:.1f}% time saved")
        
        # Analysis of preliminary FRC distribution
        preliminary_frcs = results_df['preliminary_frc'].dropna()
        if len(preliminary_frcs) > 0:
            below_threshold = (preliminary_frcs < self.frc_threshold).sum()
            above_threshold = (preliminary_frcs >= self.frc_threshold).sum()
            
            print(f"\n📈 PRELIMINARY FRC_URB2D DISTRIBUTION:")
            print(f"  < {self.frc_threshold:.0%}: {below_threshold} cells")
            print(f"  ≥ {self.frc_threshold:.0%}: {above_threshold} cells")
            print(f"  Range: [{preliminary_frcs.min():.3f}, {preliminary_frcs.max():.3f}]")
            print(f"  Mean: {preliminary_frcs.mean():.3f}")
    
    def _compare_results(self, seq_results, par_results):
        """Compare sequential vs parallel results for debugging"""
        if not seq_results or not par_results:
            print("❌ Cannot compare: missing results")
            return
        
        print(f"\n📊 RESULTS COMPARISON (first 3 cells):")
        print(f"{'Parameter':<15} {'Sequential':<12} {'Parallel':<12} {'Match':<8}")
        print("-" * 55)
        
        for i in range(min(3, len(seq_results), len(par_results))):
            seq = seq_results[i]
            par = par_results[i]
            
            for param in ['FRC_URB2D', 'BLDFR_URB2D', 'building_count']:
                seq_val = seq.get(param, 0)
                par_val = par.get(param, 0)
                if pd.isna(seq_val) and pd.isna(par_val):
                    match = "✅"
                elif pd.isna(seq_val) or pd.isna(par_val):
                    match = "❌"
                else:
                    match = "✅" if abs(seq_val - par_val) < 0.001 else "❌"
                
                seq_str = "NA" if pd.isna(seq_val) else f"{seq_val:.3f}"
                par_str = "NA" if pd.isna(par_val) else f"{par_val:.3f}"
                print(f"{param:<15} {seq_str:<12} {par_str:<12} {match:<8}")
        
        return True
    
    def _calculate_parameters_sequential(self):
        """Optimized sequential calculation with FRC_URB2D threshold"""
        
        # === PRE-PROCESSING SPATIAL JOIN ===
        print("🚀 Pre-processing: spatial join of data with grid...")
        
        # Check if unified features are available
        use_unified_features = hasattr(self, 'unified_features') and self.unified_features is not None
        
        if use_unified_features:
            print("   🔗 Using unified features dataset (with overlap resolution)")
            # Pre-join unified features with grid
            unified_with_cells = self._spatial_join_optimized(self.unified_features, 'unified features')
        else:
            print("   📋 Using separate feature datasets (fallback method)")
            # Pre-join individual datasets
            buildings_with_cells = self._spatial_join_optimized(self.buildings, 'buildings')
            
            # Pre-join roads with grid
            roads_with_cells = None
            if self.roads is not None:
                roads_with_cells = self._spatial_join_optimized(self.roads, 'roads')
            
            # Pre-join impervious surfaces with grid
            impervious_with_cells = None
            if 'impervious_surfaces' in self.other_features:
                impervious_with_cells = self._spatial_join_optimized(
                    self.other_features['impervious_surfaces'], 'impervious'
                )
        
        # === PARAMETER CALCULATION WITH THRESHOLD ===
        print(f"⚡ Sequential parameter calculation with {self.frc_threshold:.0%} threshold...")
        
        results = []
        skipped_count = 0
        calculated_count = 0
        
        # Progress bar
        for idx, cell in tqdm(self.grid.iterrows(), total=len(self.grid), desc="Processing cells"):
            try:
                # STEP 1: Fast preliminary FRC_URB2D calculation
                if use_unified_features:
                    preliminary_frc = self._calculate_preliminary_frc_unified(cell, unified_with_cells)
                else:
                    preliminary_frc = self._calculate_preliminary_frc_urb2d(
                        cell, buildings_with_cells, roads_with_cells, impervious_with_cells
                    )
                
                # STEP 2: Threshold-based decision
                if preliminary_frc >= self.frc_threshold:
                    # Complete calculation of all parameters
                    if use_unified_features:
                        params = self._calculate_cell_parameters_unified(cell, unified_with_cells)
                    else:
                        params = self._calculate_cell_parameters_optimized(
                            cell, buildings_with_cells, roads_with_cells, impervious_with_cells
                        )
                    calculated_count += 1
                    
                    if self._debug_mode:
                        print(f"    ✅ Cell {cell['cell_id']}: FRC={preliminary_frc:.3f} → full calculation")
                        
                else:
                    # Assign NA to all parameters
                    params = self._na_parameters()
                    skipped_count += 1
                    
                    if self._debug_mode:
                        print(f"    ⏭️ Cell {cell['cell_id']}: FRC={preliminary_frc:.3f} → skipped")
                
                # Add cell identifiers
                params['cell_id'] = cell['cell_id']
                params['x_index'] = cell['x_index']
                params['y_index'] = cell['y_index']
                params['preliminary_frc'] = preliminary_frc
                
                results.append(params)
                
            except Exception as e:
                if self._debug_mode:
                    print(f"❌ Error in cell {cell['cell_id']}: {e}")
                
                # Add NA parameters for error
                params = self._na_parameters()
                params['cell_id'] = cell['cell_id']
                params['x_index'] = cell['x_index'] 
                params['y_index'] = cell['y_index']
                params['preliminary_frc'] = 0.0
                results.append(params)
                skipped_count += 1
        
        print(f"📊 Threshold {self.frc_threshold:.0%} results: {calculated_count} cells calculated, {skipped_count} skipped")
        return results
    
    def _calculate_preliminary_frc_unified(self, cell, unified_with_cells):
        """Fast preliminary FRC_URB2D calculation using unified features"""
        cell_bounds = cell.geometry.bounds
        cell_area = cell.geometry.area
        
        try:
            # Filter unified features for this cell
            minx, miny, maxx, maxy = cell_bounds
            
            if 'index_right' in unified_with_cells.columns:
                cell_features = unified_with_cells[unified_with_cells['index_right'] == cell.name]
            else:
                mask = (
                    (unified_with_cells['feature_minx'] < maxx) &
                    (unified_with_cells['feature_maxx'] > minx) &
                    (unified_with_cells['feature_miny'] < maxy) &
                    (unified_with_cells['feature_maxy'] > miny)
                )
                cell_features = unified_with_cells[mask]
            
            if len(cell_features) == 0:
                return 0.0
            
            # Fast area estimation with priority
            total_estimated_area = 0.0
            
            # Sort by priority to avoid double counting
            cell_features_sorted = cell_features.sort_values('priority') if 'priority' in cell_features.columns else cell_features
            
            for idx, feature in cell_features_sorted.iterrows():
                # Fast bounds-based area estimation
                feat_bounds = feature.geometry.bounds if hasattr(feature.geometry, 'bounds') else [
                    feature.get('feature_minx', minx),
                    feature.get('feature_miny', miny), 
                    feature.get('feature_maxx', maxx),
                    feature.get('feature_maxy', maxy)
                ]
                
                # Calculate approximate overlap
                overlap_x = max(0, min(maxx, feat_bounds[2]) - max(minx, feat_bounds[0]))
                overlap_y = max(0, min(maxy, feat_bounds[3]) - max(miny, feat_bounds[1]))
                overlap_area = overlap_x * overlap_y
                
                feature_area = feature.get('feature_area', overlap_area)
                estimated_area = min(overlap_area, feature_area)
                
                total_estimated_area += estimated_area
            
            # Reduce double counting with simple factor
            if len(cell_features_sorted) > 5:
                total_estimated_area *= 0.8  # Reduction factor for overlaps
            
            preliminary_frc = min(total_estimated_area / cell_area, 1.0)
            return preliminary_frc
            
        except Exception:
            return 0.0
    
    def _calculate_cell_parameters_unified(self, cell, unified_with_cells):
        """Calculate parameters using unified features with overlap resolution"""
        cell_id = cell['cell_id']
        cell_geom = cell.geometry
        cell_area = cell_geom.area
        cell_bounds = cell_geom.bounds
        
        # === FILTER BUILDINGS FROM UNIFIED FEATURES ===
        minx, miny, maxx, maxy = cell_bounds
        
        if 'index_right' in unified_with_cells.columns:
            cell_features = unified_with_cells[unified_with_cells['index_right'] == cell.name]
        else:
            mask = (
                (unified_with_cells['feature_minx'] < maxx) &
                (unified_with_cells['feature_maxx'] > minx) &
                (unified_with_cells['feature_miny'] < maxy) &
                (unified_with_cells['feature_maxy'] > miny)
            )
            cell_features = unified_with_cells[mask]
        
        # Extract buildings for morphometric analysis
        cell_buildings = cell_features[cell_features['feature_type'] == 'building'].copy() if 'feature_type' in cell_features.columns else cell_features
        
        if len(cell_buildings) == 0:
            return self._na_parameters()
        
        # === BUILDING INTERSECTION CALCULATION ===
        building_intersections = []
        total_building_area = 0.0
        
        for idx, building in cell_buildings.iterrows():
            try:
                intersection = building.geometry.intersection(cell_geom)
                if intersection.is_empty or not intersection.is_valid:
                    continue
                
                area = intersection.area
                if area <= 0:
                    continue
                
                # Get building parameters
                height = building.get('height_m', 9.0)
                width = building.get('width_m', 15.0) if 'width_m' in building else 15.0
                original_area = building.get('area_m2', area)
                
                building_intersections.append({
                    'area': area,
                    'height': height,
                    'width': width,
                    'original_area': original_area,
                    'centroid': intersection.centroid
                })
                total_building_area += area
                
            except Exception:
                continue
        
        if not building_intersections:
            return self._na_parameters()
        
        # === PARAMETER CALCULATION ===
        heights = [b['height'] for b in building_intersections]
        areas = [b['area'] for b in building_intersections]
        
        mh_min = min(heights)
        mh_max = max(heights)
        mh_mean = np.average(heights, weights=areas)
        sigma_h = np.sqrt(np.average((np.array(heights) - mh_mean)**2, weights=areas))
        
        bldfr_urb2d = total_building_area / cell_area
        
        widths = [b['width'] for b in building_intersections]
        bw = np.average(widths, weights=areas)
        
        # Estimate street width (simplified for unified features)
        street_width = 20.0  # Default estimate when using unified features
        if len(building_intersections) > 1:
            # Simple density-based estimation
            building_density = len(building_intersections) / cell_area
            street_width = max(5.0, min(50.0, 1.0 / np.sqrt(building_density))) if building_density > 0 else 20.0
        
        h2w = mh_mean / max(street_width, 1.0) if street_width > 0 else 0.0
        
        # Calculate FRC_URB2D using unified features
        frc_urb2d = self._calculate_impervious_fraction_unified(cell_bounds, cell_area, cell.name)
        
        # Aerodynamic parameters
        lambda_p = bldfr_urb2d
        z0 = self._calculate_roughness_length(mh_mean, lambda_p)
        zd = self._calculate_displacement_height(mh_mean, lambda_p)
        
        return {
            'FRC_URB2D': frc_urb2d,
            'MH_URB2D_MIN': mh_min,
            'MH_URB2D': mh_mean,
            'MH_URB2D_MAX': mh_max,
            'BLDFR_URB2D': bldfr_urb2d,
            'H2W': h2w,
            'BW': bw,
            'Z0': z0,
            'ZD': zd,
            'Sigma': sigma_h,
            'building_count': len(building_intersections),
            'total_building_area': total_building_area,
            'street_width_analyzed': street_width,
            'lambda_p': lambda_p
        }
    
    def _calculate_preliminary_frc_urb2d(self, cell, buildings_with_cells, roads_with_cells, impervious_with_cells):
        """
        Fast preliminary FRC_URB2D calculation to decide whether to proceed
        """
        cell_bounds = cell.geometry.bounds
        cell_area = cell.geometry.area
        
        try:
            # === FAST BUILDING FILTERING ===
            cell_buildings = _filter_buildings_for_cell(buildings_with_cells, cell.name, cell_bounds)
            
            if len(cell_buildings) == 0:
                return 0.0
            
            # === FAST BUILDING AREA ESTIMATION ===
            building_area_estimate = 0.0
            
            for idx, building in cell_buildings.iterrows():
                # Fast estimation: use original area if centroid inside cell
                # or fraction based on bounds overlap
                building_bounds = building.geometry.bounds
                
                # Calculate approximate overlap based on bounds
                overlap_x = max(0, min(cell_bounds[2], building_bounds[2]) - max(cell_bounds[0], building_bounds[0]))
                overlap_y = max(0, min(cell_bounds[3], building_bounds[3]) - max(cell_bounds[1], building_bounds[1]))
                overlap_area = overlap_x * overlap_y
                
                # Use actual building area limited by overlap
                building_area = building.get('area_m2', overlap_area)
                estimated_area = min(overlap_area, building_area)
                
                building_area_estimate += estimated_area
            
            # === FAST ROAD AREA ESTIMATION ===
            road_area_estimate = 0.0
            
            if roads_with_cells is not None:
                minx, miny, maxx, maxy = cell_bounds
                
                # Filter roads quickly
                if 'index_right' in roads_with_cells.columns:
                    cell_roads = roads_with_cells[roads_with_cells['index_right'] == cell.name]
                else:
                    mask = (
                        (roads_with_cells['feature_minx'] < maxx) &
                        (roads_with_cells['feature_maxx'] > minx) &
                        (roads_with_cells['feature_miny'] < maxy) &
                        (roads_with_cells['feature_maxy'] > miny)
                    )
                    cell_roads = roads_with_cells[mask]
                
                # Estimate road area
                for idx, road in cell_roads.iterrows():
                    road_width = road.get('road_width_m', 5.0)
                    road_length_in_cell = min(road.geometry.length, self.grid_size_m)  # Approximation
                    road_area_estimate += road_length_in_cell * road_width * 0.7  # Reduction factor
            
            # === PRELIMINARY FRC CALCULATION ===
            total_impervious_estimate = building_area_estimate + road_area_estimate
            preliminary_frc = min(total_impervious_estimate / cell_area, 1.0)
            
            return preliminary_frc
            
        except Exception:
            return 0.0
    
    def _calculate_parameters_parallel(self):
        """Parallel calculation with FRC_URB2D threshold"""
        try:
            # === PRE-PROCESSING DATA FOR MULTIPROCESSING ===
            print("📋 Preparing data for parallel calculation with threshold...")
            
            # Pre-process buildings
            buildings_with_cells = self._spatial_join_optimized(self.buildings, 'buildings')
            
            # Pre-process roads  
            roads_with_cells = None
            if self.roads is not None:
                roads_with_cells = self._spatial_join_optimized(self.roads, 'roads')
            
            # Pre-process impervious surfaces
            impervious_with_cells = None
            if 'impervious_surfaces' in self.other_features:
                impervious_with_cells = self._spatial_join_optimized(
                    self.other_features['impervious_surfaces'], 'impervious'
                )
            
            # === SERIALIZE DATA FOR MULTIPROCESSING ===
            def serialize_geodf(gdf, name):
                if gdf is None or len(gdf) == 0:
                    return None
                return {
                    'data': gdf.to_dict('records'),
                    'crs': gdf.crs
                }
            
            buildings_data = serialize_geodf(buildings_with_cells, 'buildings')
            roads_data = serialize_geodf(roads_with_cells, 'roads') 
            impervious_data = serialize_geodf(impervious_with_cells, 'impervious surfaces')
            
            # Serialize unified features if available
            unified_features_data = None
            if hasattr(self, 'unified_features') and self.unified_features is not None:
                unified_features_data = serialize_geodf(self.unified_features, 'unified features')
                print(f"📋 Unified features serialized: {len(self.unified_features)} features")
            
            # === PREPARE ARGUMENTS FOR WORKER WITH THRESHOLD ===
            worker_args = []
            
            for idx, cell in self.grid.iterrows():
                cell_data = {
                    'geometry': cell.geometry,
                    'properties': {
                        'cell_id': cell['cell_id'],
                        'x_index': cell['x_index'],
                        'y_index': cell['y_index']
                    },
                    'crs': self.grid.crs,
                    'index': idx,  # Original index for spatial join lookup
                    'frc_threshold': self.frc_threshold  # ADDED: threshold for worker
                }
                
                # Add unified features data if available
                if unified_features_data is not None:
                    cell_data['unified_features_data'] = unified_features_data
                
                worker_args.append((cell_data, buildings_data, roads_data, impervious_data))
            
            # === PARALLEL EXECUTION ===
            print(f"⚡ Starting {self.n_cores} worker processes with {self.frc_threshold:.0%} threshold...")
            
            results = []
            
            # Use ProcessPoolExecutor for better control
            with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
                # Submit all tasks
                future_to_cell = {
                    executor.submit(_parallel_cell_worker, args): i 
                    for i, args in enumerate(worker_args)
                }
                
                # Collect results with progress bar
                with tqdm(total=len(worker_args), desc="Parallel parameter calculation with threshold") as pbar:
                    for future in as_completed(future_to_cell):
                        try:
                            result = future.result(timeout=30)  # 30s timeout per cell
                            results.append(result)
                        except Exception as e:
                            cell_idx = future_to_cell[future]
                            # Add NA parameters as fallback
                            fallback = _na_parameters_static()
                            fallback.update({
                                'cell_id': f'error_{cell_idx}',
                                'x_index': 0,
                                'y_index': 0,
                                'preliminary_frc': 0.0
                            })
                            results.append(fallback)
                        finally:
                            pbar.update(1)
            
            print(f"✅ Parallel calculation with threshold completed: {len(results)} cells processed")
            return results
            
        except Exception as e:
            print(f"❌ Error in parallel calculation: {e}")
            print("🔄 Fallback to sequential calculation...")
            return self._calculate_parameters_sequential()
    
    def _spatial_join_optimized(self, gdf, layer_name):
        """
        Pre-processing: optimized spatial join to assign features to cells
        """
        if self._debug_mode:
            print(f"  📊 Processing {layer_name}: {len(gdf)} features...")
        
        # Add bounding box coordinates to cells for fast filtering
        self.grid['minx'] = self.grid.geometry.bounds['minx']
        self.grid['miny'] = self.grid.geometry.bounds['miny'] 
        self.grid['maxx'] = self.grid.geometry.bounds['maxx']
        self.grid['maxy'] = self.grid.geometry.bounds['maxy']
        
        # Add bounding box coordinates to features
        gdf_bounds = gdf.bounds
        gdf['feature_minx'] = gdf_bounds['minx']
        gdf['feature_miny'] = gdf_bounds['miny']
        gdf['feature_maxx'] = gdf_bounds['maxx']
        gdf['feature_maxy'] = gdf_bounds['maxy']
        
        # Spatial join using spatial index (much faster)
        try:
            joined = gpd.sjoin(gdf, self.grid, how='left', predicate='intersects')
            if self._debug_mode:
                print(f"    ✓ {len(joined)} {layer_name}-cell associations")
            return joined
        except Exception as e:
            if self._debug_mode:
                print(f"    ⚠️ Fallback to slower method for {layer_name}: {e}")
            # Fallback to original method if sjoin fails
            return gdf
    
    def _calculate_cell_parameters_optimized(self, cell, buildings_with_cells, roads_with_cells, impervious_with_cells):
        """
        Calculate parameters for a cell using pre-processed data
        UNIFIED VERSION that must match parallel worker
        """
        cell_id = cell['cell_id']
        cell_geom = cell.geometry
        cell_area = cell_geom.area
        cell_bounds = cell_geom.bounds
        
        # === UNIFIED BUILDING FILTERING ===
        cell_buildings = _filter_buildings_for_cell(buildings_with_cells, cell.name, cell_bounds)
        
        if self._debug_mode and len(cell_buildings) > 0:
            print(f"    🔍 Cell {cell_id}: {len(cell_buildings)} candidate buildings")
            
        if len(cell_buildings) == 0:
            return self._na_parameters()
        
        # === PRECISE INTERSECTION CALCULATION ===
        building_intersections = []
        total_building_area = 0.0
        split_buildings_count = 0
        
        for idx, building in cell_buildings.iterrows():
            try:
                # Precise geometric intersection
                intersection = building.geometry.intersection(cell_geom)
                if intersection.is_empty or not intersection.is_valid:
                    continue
                
                area = intersection.area
                if area <= 0:
                    continue
                
                # Verify geometric consistency
                original_area = building.get('area_m2', area)
                if area > original_area * 1.01:  # 1% tolerance
                    area = min(area, original_area)
                
                intersection_fraction = area / original_area if original_area > 0 else 1.0
                
                # Determine if building is straddling (IMPROVED LOGIC)
                crossing_info = _analyze_building_cell_relationship(
                    building.geometry.bounds, cell_bounds, area, original_area
                )
                
                is_straddling = crossing_info['is_straddling']
                if is_straddling:
                    split_buildings_count += 1
                
                building_intersections.append({
                    'area': area,
                    'height': building.get('height_m', 9.0),
                    'width': building.get('width_m', 15.0),
                    'original_area': original_area,
                    'intersection_fraction': intersection_fraction,
                    'centroid': intersection.centroid,
                    'is_straddling': is_straddling,
                    'crossing_type': crossing_info['crossing_type']
                })
                total_building_area += area
                
            except Exception as e:
                if self._debug_mode:
                    print(f"    ❌ Building intersection error {idx}: {e}")
                continue
        
        if self._debug_mode and building_intersections:
            print(f"    ✅ Cell {cell_id}: {len(building_intersections)} actual buildings, "
                  f"total area: {total_building_area:.1f} m² ({total_building_area/cell_area:.2%})")
        
        if not building_intersections:
            return self._na_parameters()
        
        # === PARAMETER CALCULATION ===
        
        # 1. Building height parameters (weighted by area in cell)
        heights = [b['height'] for b in building_intersections]
        areas = [b['area'] for b in building_intersections]
        
        mh_min = min(heights)
        mh_max = max(heights)
        mh_mean = np.average(heights, weights=areas)
        sigma_h = np.sqrt(np.average((np.array(heights) - mh_mean)**2, weights=areas))
        
        # 2. Building surface fraction (BLDFR_URB2D / PAI)
        bldfr_urb2d = total_building_area / cell_area
        
        # 3. Average building width (weighted by actual area)
        widths = [b['width'] for b in building_intersections]
        bw = np.average(widths, weights=areas)
        
        # 4. Inter-building distance analysis using roads (optimized)
        street_width = self._analyze_building_distances_optimized(
            cell_bounds, building_intersections, roads_with_cells, cell.name
        )
        
        # 5. Height to Width ratio
        h2w = mh_mean / max(street_width, 1.0) if street_width > 0 else 0.0
        
        # 6. Urban impervious fraction (FRC_URB2D) - using unified features with overlap resolution
        if hasattr(self, 'unified_features') and self.unified_features is not None:
            frc_urb2d = self._calculate_impervious_fraction_unified(cell_bounds, cell_area, cell.name)
        else:
            # Fallback to optimized method if unified features not available
            frc_urb2d = self._calculate_impervious_fraction_optimized(
                cell_bounds, cell_area, total_building_area, cell_buildings, 
                roads_with_cells, impervious_with_cells, cell.name
            )
        
        # 7. Aerodynamic parameters
        lambda_p = bldfr_urb2d
        z0 = self._calculate_roughness_length(mh_mean, lambda_p)
        zd = self._calculate_displacement_height(mh_mean, lambda_p)
        
        # === QUALITY CHECKS ===
        if frc_urb2d < bldfr_urb2d and self._debug_mode:
            print(f"    ⚠️ WARNING: FRC_URB2D ({frc_urb2d:.3f}) < BLDFR_URB2D ({bldfr_urb2d:.3f})")
        
        return {
            'FRC_URB2D': frc_urb2d,
            'MH_URB2D_MIN': mh_min,
            'MH_URB2D': mh_mean,
            'MH_URB2D_MAX': mh_max,
            'BLDFR_URB2D': bldfr_urb2d,
            'H2W': h2w,
            'BW': bw,
            'Z0': z0,
            'ZD': zd,
            'Sigma': sigma_h,
            # Auxiliary parameters with additional geometric information
            'building_count': len(building_intersections),
            'total_building_area': total_building_area,
            'street_width_analyzed': street_width,
            'lambda_p': lambda_p
        }
    
    def _analyze_building_distances_optimized(self, cell_bounds, building_intersections, roads_with_cells, cell_index):
        """Optimized building-road distance analysis using pre-filtering"""
        if roads_with_cells is None or len(building_intersections) < 2:
            return 20.0
        
        try:
            # Filter roads for this cell (much faster)
            minx, miny, maxx, maxy = cell_bounds
            
            if 'index_right' in roads_with_cells.columns:
                # Use spatial join results
                cell_roads = roads_with_cells[
                    roads_with_cells['index_right'] == cell_index
                ]
            else:
                # Coordinate filtering
                mask = (
                    (roads_with_cells['feature_minx'] < maxx) &
                    (roads_with_cells['feature_maxx'] > minx) &
                    (roads_with_cells['feature_miny'] < maxy) &
                    (roads_with_cells['feature_maxy'] > miny)
                )
                cell_roads = roads_with_cells[mask]
            
            if len(cell_roads) == 0:
                # Fallback: estimate from building density
                cell_area = (maxx - minx) * (maxy - miny)
                building_density = len(building_intersections) / cell_area
                return max(5.0, min(50.0, 1.0 / np.sqrt(building_density))) if building_density > 0 else 20.0
            
            # Calculate distances (unchanged logic but on filtered dataset)
            building_centroids = [b['centroid'] for b in building_intersections]
            road_distances = []
            
            for centroid in building_centroids:
                min_distance = float('inf')
                for idx, road in cell_roads.iterrows():
                    try:
                        distance = centroid.distance(road.geometry)
                        if distance < min_distance:
                            min_distance = distance
                    except:
                        continue
                
                if min_distance != float('inf'):
                    road_distances.append(min_distance)
            
            if road_distances:
                mean_distance_to_road = np.mean(road_distances)
                street_width = mean_distance_to_road * 2.0
                
                road_widths = cell_roads['road_width_m'].tolist()
                if road_widths:
                    physical_road_width = np.mean(road_widths)
                    street_width = max(street_width, physical_road_width)
                
                return max(3.0, min(street_width, 50.0))
            else:
                return 15.0
                
        except Exception:
            return 15.0
    
    def _calculate_impervious_fraction_unified(self, cell_bounds, cell_area, cell_index):
        """
        Calculate optimized impervious fraction using unified features with priority-based overlap resolution
        """
        try:
            minx, miny, maxx, maxy = cell_bounds
            cell_geom = box(minx, miny, maxx, maxy)
            
            # === 1. FILTER UNIFIED FEATURES FOR THIS CELL ===
            if hasattr(self, 'unified_features') and self.unified_features is not None:
                # Fast coordinate-based filtering
                mask = (
                    (self.unified_features['feature_minx'] < maxx) &
                    (self.unified_features['feature_maxx'] > minx) &
                    (self.unified_features['feature_miny'] < maxy) &
                    (self.unified_features['feature_maxy'] > miny)
                )
                cell_features = self.unified_features[mask].copy()
            else:
                # Fallback to old method if unified features not available
                return self._calculate_impervious_fraction_optimized(
                    cell_bounds, cell_area, 0, None, None, None, cell_index
                )
            
            if len(cell_features) == 0:
                return 0.0
            
            # === 2. INTERSECT FEATURES WITH CELL ===
            intersected_features = []
            
            for idx, feature in cell_features.iterrows():
                try:
                    intersection = feature.geometry.intersection(cell_geom)
                    if not intersection.is_empty and intersection.is_valid and intersection.area > 0:
                        intersected_features.append({
                            'geometry': intersection,
                            'priority': feature['priority'],
                            'feature_type': feature['feature_type'],
                            'area': intersection.area
                        })
                except Exception:
                    continue
            
            if not intersected_features:
                return 0.0
            
            # === 3. PRIORITY-BASED OVERLAP RESOLUTION ===
            # Sort by priority (1=highest, 3=lowest)
            intersected_features.sort(key=lambda x: x['priority'])
            
            # Process features by priority to avoid overlaps
            remaining_cell_area = cell_geom
            total_impervious_area = 0.0
            
            for feature in intersected_features:
                if remaining_cell_area.is_empty or remaining_cell_area.area <= 0:
                    break
                
                try:
                    # Calculate intersection with remaining area
                    effective_intersection = feature['geometry'].intersection(remaining_cell_area)
                    
                    if not effective_intersection.is_empty and effective_intersection.area > 0:
                        # Add to total impervious area
                        area_to_add = effective_intersection.area
                        total_impervious_area += area_to_add
                        
                        # Remove this area from remaining cell area to prevent overlaps
                        remaining_cell_area = remaining_cell_area.difference(effective_intersection)
                        
                        if self._debug_mode:
                            print(f"        {feature['feature_type']} (priority {feature['priority']}): +{area_to_add:.1f} m²")
                
                except Exception as e:
                    if self._debug_mode:
                        print(f"        Error processing {feature['feature_type']}: {e}")
                    continue
            
            # === 4. CALCULATE FINAL FRACTION ===
            frc_urb2d = min(total_impervious_area / cell_area, 1.0)
            
            if self._debug_mode:
                print(f"        Total impervious: {total_impervious_area:.1f} m² / {cell_area:.1f} m² = {frc_urb2d:.3f}")
            
            return frc_urb2d
            
        except Exception as e:
            if self._debug_mode:
                print(f"        Error in unified impervious calculation: {e}")
            return 0.0
    
    def _calculate_impervious_fraction_optimized(self, cell_bounds, cell_area, building_area, 
                                               cell_buildings, roads_with_cells, impervious_with_cells, cell_index):
        """Calculate optimized impervious fraction using pre-filtering"""
        try:
            all_impervious_geometries = []
            minx, miny, maxx, maxy = cell_bounds
            cell_geom = box(minx, miny, maxx, maxy)
            
            # === 1. BUILDINGS (already filtered) ===
            if building_area > 0 and cell_buildings is not None:
                for idx, building in cell_buildings.iterrows():
                    try:
                        intersection = building.geometry.intersection(cell_geom)
                        if not intersection.is_empty and intersection.is_valid:
                            all_impervious_geometries.append(intersection)
                    except:
                        continue
            
            # === 2. ROADS (filtered) ===
            if roads_with_cells is not None:
                if 'index_right' in roads_with_cells.columns:
                    cell_roads = roads_with_cells[roads_with_cells['index_right'] == cell_index]
                else:
                    mask = (
                        (roads_with_cells['feature_minx'] < maxx) &
                        (roads_with_cells['feature_maxx'] > minx) &
                        (roads_with_cells['feature_miny'] < maxy) &
                        (roads_with_cells['feature_maxy'] > miny)
                    )
                    cell_roads = roads_with_cells[mask]
                
                for idx, road in cell_roads.iterrows():
                    try:
                        road_width = road.get('road_width_m', 5.0)
                        buffered_road = road.geometry.buffer(road_width / 2.0)
                        road_intersection = buffered_road.intersection(cell_geom)
                        
                        if not road_intersection.is_empty and road_intersection.is_valid:
                            all_impervious_geometries.append(road_intersection)
                    except:
                        continue
            
            # === 3. OTHER IMPERVIOUS SURFACES (filtered) ===
            if impervious_with_cells is not None:
                if 'index_right' in impervious_with_cells.columns:
                    cell_impervious = impervious_with_cells[impervious_with_cells['index_right'] == cell_index]
                else:
                    mask = (
                        (impervious_with_cells['feature_minx'] < maxx) &
                        (impervious_with_cells['feature_maxx'] > minx) &
                        (impervious_with_cells['feature_miny'] < maxy) &
                        (impervious_with_cells['feature_maxy'] > miny)
                    )
                    cell_impervious = impervious_with_cells[mask]
                
                for idx, surface in cell_impervious.iterrows():
                    try:
                        intersection = surface.geometry.intersection(cell_geom)
                        if not intersection.is_empty and intersection.is_valid:
                            all_impervious_geometries.append(intersection)
                    except:
                        continue
            
            # === 4. UNIFY GEOMETRIES ===
            if all_impervious_geometries:
                try:
                    unified_impervious = unary_union(all_impervious_geometries)
                    
                    if hasattr(unified_impervious, 'area'):
                        total_impervious_area = unified_impervious.area
                    else:
                        total_impervious_area = 0.0
                    
                    frc_urb2d = min(total_impervious_area / cell_area, 1.0)
                    return frc_urb2d
                    
                except Exception:
                    # Fallback: sum areas
                    total_area = sum(geom.area for geom in all_impervious_geometries if hasattr(geom, 'area'))
                    return min(total_area / cell_area, 1.0)
            else:
                return 0.0
            
        except Exception:
            # Extreme fallback: buildings only
            return min(building_area / cell_area, 1.0) if building_area > 0 else 0.0
    
    def _calculate_roughness_length(self, mean_height, lambda_p):
        """Roughness length (Grimmond & Oke 1999 method)"""
        if lambda_p <= 0 or mean_height <= 0:
            return 0.03
        
        if lambda_p < 0.15:
            z0 = mean_height * (0.1 * lambda_p)
        elif lambda_p < 0.35:
            z0 = mean_height * (0.05 + 0.05 * lambda_p / 0.35)
        else:
            z0 = mean_height * 0.1
        
        return max(min(z0, mean_height * 0.2), 0.1)
    
    def _calculate_displacement_height(self, mean_height, lambda_p):
        """Zero-plane displacement height"""
        if lambda_p <= 0 or mean_height <= 0:
            return 0.0
        
        if lambda_p < 0.15:
            zd = mean_height * (3.0 * lambda_p)
        elif lambda_p < 0.35:
            zd = mean_height * (0.45 + 0.3 * (lambda_p - 0.15) / 0.2)
        else:
            zd = mean_height * 0.75
        
        return max(min(zd, mean_height * 0.8), 0.0)
    
    def _na_parameters(self):
        """NA parameters for cells without buildings or below threshold"""
        return {
            'FRC_URB2D': np.nan,
            'MH_URB2D_MIN': np.nan,
            'MH_URB2D': np.nan,
            'MH_URB2D_MAX': np.nan,
            'BLDFR_URB2D': np.nan,
            'H2W': np.nan,
            'BW': np.nan,
            'Z0': np.nan,
            'ZD': np.nan,
            'Sigma': np.nan,
            'building_count': 0,
            'total_building_area': 0.0,
            'street_width_analyzed': np.nan,
            'lambda_p': np.nan
        }
    
    # === FILE MANAGEMENT METHODS ===
    
    def _clean_buildings_for_save(self):
        """Clean buildings data before saving to avoid SQLite/GPKG issues"""
        buildings_clean = self.buildings.copy()
        
        try:
            # 1. Keep only essential columns to avoid SQLite reserved words issues
            essential_columns = ['geometry', 'height_m', 'height_source', 'area_m2', 'perimeter_m', 'width_m', 'compactness']
            
            # Check which essential columns exist
            available_columns = ['geometry']  # geometry is always required
            for col in essential_columns[1:]:  # skip geometry
                if col in buildings_clean.columns:
                    available_columns.append(col)
            
            buildings_clean = buildings_clean[available_columns]
            
            # 2. Clean column names (remove special characters)
            column_mapping = {}
            for col in buildings_clean.columns:
                if col != 'geometry':
                    # Replace problematic characters
                    clean_col = col.replace(':', '_').replace('-', '_').replace(' ', '_')
                    # Ensure it doesn't start with number
                    if clean_col[0].isdigit():
                        clean_col = f"col_{clean_col}"
                    column_mapping[col] = clean_col
            
            if column_mapping:
                buildings_clean = buildings_clean.rename(columns=column_mapping)
            
            # 3. Ensure geometries are valid
            buildings_clean = buildings_clean[buildings_clean.geometry.is_valid]
            
            # 4. Remove any infinite or extremely large values
            for col in buildings_clean.columns:
                if col != 'geometry' and buildings_clean[col].dtype in ['float64', 'int64']:
                    buildings_clean = buildings_clean[
                        (buildings_clean[col] >= -1e6) & 
                        (buildings_clean[col] <= 1e6) &
                        buildings_clean[col].notna()
                    ]
            
            # 5. Limit to first 500k buildings if too many (memory management)
            if len(buildings_clean) > 500000:
                print(f"⚠️ Limiting to first 500k buildings (original: {len(buildings_clean)})")
                buildings_clean = buildings_clean.head(500000)
            
            return buildings_clean
            
        except Exception as e:
            print(f"⚠️ Error cleaning buildings data: {e}")
            # Return original data as last resort
            return self.buildings
    
    def _clean_roads_for_save(self):
        """Clean roads data before saving to avoid SQLite/GPKG issues"""
        roads_clean = self.roads.copy()
        
        try:
            # Keep only essential columns
            essential_columns = ['geometry', 'highway', 'road_hierarchy', 'road_width_m']
            available_columns = ['geometry']
            
            for col in essential_columns[1:]:
                if col in roads_clean.columns:
                    available_columns.append(col)
            
            roads_clean = roads_clean[available_columns]
            
            # Clean column names
            column_mapping = {}
            for col in roads_clean.columns:
                if col != 'geometry':
                    clean_col = col.replace(':', '_').replace('-', '_').replace(' ', '_')
                    if clean_col[0].isdigit():
                        clean_col = f"col_{clean_col}"
                    column_mapping[col] = clean_col
            
            if column_mapping:
                roads_clean = roads_clean.rename(columns=column_mapping)
            
            # Ensure geometries are valid
            roads_clean = roads_clean[roads_clean.geometry.is_valid]
            
            # Remove extremely long roads (likely errors)
            if 'geometry' in roads_clean.columns:
                road_lengths = roads_clean.geometry.length
                roads_clean = roads_clean[road_lengths <= 50000]  # Max 50km per segment
            
            # Limit number of roads if too many
            if len(roads_clean) > 200000:
                print(f"⚠️ Limiting to first 200k roads (original: {len(roads_clean)})")
                roads_clean = roads_clean.head(200000)
            
            return roads_clean
            
        except Exception as e:
            print(f"⚠️ Error cleaning roads data: {e}")
            return self.roads
    
    def _clean_impervious_for_save(self, impervious_gdf):
        """Clean impervious surfaces data before saving to avoid SQLite/GPKG issues"""
        try:
            # Keep only essential columns
            essential_columns = ['geometry', 'surface_type']
            available_columns = ['geometry']
            
            for col in essential_columns[1:]:
                if col in impervious_gdf.columns:
                    available_columns.append(col)
            
            impervious_clean = impervious_gdf[available_columns].copy()
            
            # Clean column names
            column_mapping = {}
            for col in impervious_clean.columns:
                if col != 'geometry':
                    clean_col = col.replace(':', '_').replace('-', '_').replace(' ', '_')
                    if clean_col[0].isdigit():
                        clean_col = f"col_{clean_col}"
                    column_mapping[col] = clean_col
            
            if column_mapping:
                impervious_clean = impervious_clean.rename(columns=column_mapping)
            
            # Ensure geometries are valid
            impervious_clean = impervious_clean[impervious_clean.geometry.is_valid]
            
            # Remove extremely large polygons (likely errors)
            if len(impervious_clean) > 0:
                areas = impervious_clean.geometry.area
                # Remove polygons larger than 10 km²
                impervious_clean = impervious_clean[areas <= 10_000_000]
            
            # Limit number if too many
            if len(impervious_clean) > 100000:
                print(f"⚠️ Limiting to first 100k impervious surfaces (original: {len(impervious_clean)})")
                impervious_clean = impervious_clean.head(100000)
            
            return impervious_clean
            
        except Exception as e:
            print(f"⚠️ Error cleaning impervious data: {e}")
            return impervious_gdf
    
    def _safe_file_save(self, gdf, filepath, data_type="data"):
        """Safe file saving with multiple fallback strategies"""
        try:
            # Try GPKG first (preferred)
            gdf.to_file(filepath, driver='GPKG')
            return True
        except Exception as e1:
            print(f"⚠️ GPKG save failed for {data_type}: {e1}")
            
            try:
                # Try Shapefile as fallback
                shp_path = filepath.replace('.gpkg', '.shp')
                gdf.to_file(shp_path, driver='ESRI Shapefile')
                print(f"✅ Saved as Shapefile instead: {shp_path}")
                return True
            except Exception as e2:
                print(f"⚠️ Shapefile save also failed for {data_type}: {e2}")
                
                try:
                    # Try GeoJSON as last resort
                    json_path = filepath.replace('.gpkg', '.geojson')
                    gdf.to_file(json_path, driver='GeoJSON')
                    print(f"✅ Saved as GeoJSON instead: {json_path}")
                    return True
                except Exception as e3:
                    print(f"❌ All save formats failed for {data_type}: {e3}")
                    return False
    
    def _save_buildings_temp_file(self, buildings_file):
        """Save buildings temporary file with robust error handling"""
        if self._debug_mode:
            print(f"💾 Attempting to save temporary buildings file: {buildings_file}")
        
        try:
            # Clean data before saving to avoid SQLite issues
            buildings_clean = self._clean_buildings_for_save()
            
            # Try to save with multiple format fallbacks
            success = self._safe_file_save(buildings_clean, buildings_file, "buildings")
            
            if success:
                if self._debug_mode:
                    print(f"✅ Buildings temporary file saved successfully")
            else:
                print(f"⚠️ Warning: Could not save buildings temporary file")
                print(f"   Continuing without temporary file (will re-download next time)")
                
        except Exception as e:
            print(f"⚠️ Unexpected error saving buildings file: {e}")
            print(f"   Continuing without temporary file")
    
    def _save_roads_temp_file(self, roads_file):
        """Save roads temporary file with robust error handling"""
        if self._debug_mode:
            print(f"💾 Attempting to save temporary roads file: {roads_file}")
        
        try:
            # Clean roads data before saving
            roads_clean = self._clean_roads_for_save()
            
            # Try to save with multiple format fallbacks
            success = self._safe_file_save(roads_clean, roads_file, "roads")
            
            if success:
                if self._debug_mode:
                    print(f"✅ Roads temporary file saved successfully")
            else:
                print(f"⚠️ Warning: Could not save roads temporary file")
                print(f"   Continuing without temporary file (will re-download next time)")
                
        except Exception as e:
            print(f"⚠️ Unexpected error saving roads file: {e}")
            print(f"   Continuing without temporary file")
    
    def _save_impervious_temp_file(self, impervious_gdf, impervious_file):
        """Save impervious surfaces temporary file with robust error handling"""
        if self._debug_mode:
            print(f"💾 Attempting to save temporary impervious surfaces file: {impervious_file}")
        
        try:
            # Clean data before saving
            impervious_clean = self._clean_impervious_for_save(impervious_gdf)
            
            # Try to save with multiple format fallbacks
            success = self._safe_file_save(impervious_clean, impervious_file, "impervious surfaces")
            
            if success:
                if self._debug_mode:
                    print(f"✅ Impervious surfaces temporary file saved successfully")
            else:
                print(f"⚠️ Warning: Could not save impervious surfaces temporary file")
                print(f"   Continuing without temporary file (will re-download next time)")
                
        except Exception as e:
            print(f"⚠️ Unexpected error saving impervious surfaces file: {e}")
            print(f"   Continuing without temporary file")
    
    def _load_temp_file_safe(self, filepath, data_type="data"):
        """Safely load temporary file with multiple format support"""
        if not os.path.exists(filepath):
            return None
        
        try:
            # Try GPKG first
            return gpd.read_file(filepath)
        except Exception as e1:
            if self._debug_mode:
                print(f"⚠️ Could not load {filepath} as GPKG: {e1}")
            
            # Try alternative formats
            for ext, driver in [('.shp', None), ('.geojson', None)]:
                alt_path = filepath.replace('.gpkg', ext)
                if os.path.exists(alt_path):
                    try:
                        return gpd.read_file(alt_path)
                    except Exception as e2:
                        if self._debug_mode:
                            print(f"⚠️ Could not load {alt_path}: {e2}")
                        continue
            
            print(f"❌ Could not load any temporary file for {data_type}")
            return None
    
    # === RESULTS SAVING METHODS ===
    
    def save_results_for_wrf(self, output_csv_path=None, include_na_cells=True):
        """
        Save results in CSV format compatible with WRF w2w workflow
        
        Args:
            output_csv_path (str): CSV file path (if None, auto-generated)
            include_na_cells (bool): Include cells with NA in CSV
        
        Returns:
            str: Generated CSV file path
        """
        if self.results is None:
            print("❌ No results to save")
            return None
        
        # Auto-generate filename if not specified
        if output_csv_path is None:
            city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
            output_csv_path = os.path.join(
                self.output_dir, 
                f"LCZ_UCP_{city_clean}_{self.grid_size_m}m_smart_buffer.csv"
            )
        
        # Prepare data for WRF CSV
        csv_data = self.results.copy()
        
        # Add required columns for WRF
        csv_data['lon'] = csv_data.geometry.centroid.x
        csv_data['lat'] = csv_data.geometry.centroid.y
        
        # Convert coordinates to WGS84 for lon/lat
        if self.results.crs != 'EPSG:4326':
            coords_wgs84 = csv_data.geometry.centroid.to_crs('EPSG:4326')
            csv_data['lon'] = coords_wgs84.x
            csv_data['lat'] = coords_wgs84.y
        
        # Columns for WRF CSV (order important)
        wrf_columns = [
            'cell_id', 'x_index', 'y_index', 'lon', 'lat',
            'FRC_URB2D', 'MH_URB2D_MIN', 'MH_URB2D', 'MH_URB2D_MAX', 
            'BLDFR_URB2D', 'H2W', 'BW'
            # Z0, ZD, Sigma are optional for WRF
        ]
        
        # Filter/order columns
        csv_for_wrf = csv_data[wrf_columns].copy()
        
        # Handle NA cells
        if not include_na_cells:
            # Remove cells with NA in FRC_URB2D
            csv_for_wrf = csv_for_wrf.dropna(subset=['FRC_URB2D'])
            print(f"  ⚠️ Removed {len(csv_data) - len(csv_for_wrf)} cells with NA")
        
        # Save CSV
        csv_for_wrf.to_csv(output_csv_path, index=False, na_rep='NA')
        
        # Final statistics
        total_cells = len(csv_for_wrf)
        urban_cells = csv_for_wrf['FRC_URB2D'].notna().sum()
        na_cells = total_cells - urban_cells
        
        print(f"💾 WRF CSV file saved: {output_csv_path}")
        print(f"  📊 Total cells: {total_cells}")
        print(f"  🏙️ Urban cells: {urban_cells} ({(urban_cells/total_cells)*100:.1f}%)")
        if na_cells > 0:
            print(f"  🌿 NA cells: {na_cells} ({(na_cells/total_cells)*100:.1f}%)")
        
        return output_csv_path
    
    def save_results(self):
        """Save all results"""
        if self.results is None:
            print("❌ No results to save")
            return None
        
        city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
        
        # Main WRF file
        wrf_columns = ['geometry', 'FRC_URB2D', 'MH_URB2D_MIN', 'MH_URB2D', 'MH_URB2D_MAX', 
                      'BLDFR_URB2D', 'H2W', 'BW', 'Z0', 'ZD', 'Sigma']
        
        wrf_file = os.path.join(self.output_dir, f"urban_parameters_{city_clean}_{self.grid_size_m}m_smart_buffer.gpkg")
        results_wrf = self.results[wrf_columns].copy()
        
        try:
            results_wrf.to_file(wrf_file, driver='GPKG')
        except Exception:
            # Fallback to shapefile
            wrf_file = wrf_file.replace('.gpkg', '.shp')
            results_wrf.to_file(wrf_file, driver='ESRI Shapefile')
        
        # Detailed file
        detailed_file = os.path.join(self.output_dir, f"detailed_parameters_{city_clean}_{self.grid_size_m}m_smart_buffer.gpkg")
        try:
            self.results.to_file(detailed_file, driver='GPKG')
        except Exception:
            # Fallback to shapefile
            detailed_file = detailed_file.replace('.gpkg', '.shp')
            self.results.to_file(detailed_file, driver='ESRI Shapefile')
        
        # CSV for WRF workflow
        csv_file = self.save_results_for_wrf()
        
        # Save analysis boundary
        boundary_file = os.path.join(self.output_dir, f"analysis_boundary_{city_clean}_smart_buffer.gpkg")
        if self.analysis_boundary is not None:
            try:
                self.analysis_boundary.to_file(boundary_file, driver='GPKG')
            except Exception:
                boundary_file = boundary_file.replace('.gpkg', '.shp')
                self.analysis_boundary.to_file(boundary_file, driver='ESRI Shapefile')
        
        print(f"💾 Results saved:")
        print(f"  - WRF GPKG: {os.path.basename(wrf_file)}")
        print(f"  - Detailed: {os.path.basename(detailed_file)}")
        print(f"  - WRF CSV: {os.path.basename(csv_file)}")
        print(f"  - Analysis boundary: {os.path.basename(boundary_file)}")
        
        return wrf_file, detailed_file, csv_file
    
    def clean_temp_files(self):
        """Clean temporary files"""
        city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
        
        temp_files = [
            os.path.join(self.output_dir, f"temp_buildings_{city_clean}.gpkg"),
            os.path.join(self.output_dir, f"temp_roads_{city_clean}.gpkg"),
            os.path.join(self.output_dir, f"temp_impervious_{city_clean}.gpkg")
        ]
        
        # Also check for alternative formats
        for base_file in temp_files[:]:
            for ext in ['.shp', '.geojson']:
                alt_file = base_file.replace('.gpkg', ext)
                if alt_file not in temp_files:
                    temp_files.append(alt_file)
        
        removed_count = 0
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    removed_count += 1
                    if self._debug_mode:
                        print(f"🗑️ Removed: {temp_file}")
                except Exception as e:
                    print(f"⚠️ Error removing {temp_file}: {e}")
        
        if removed_count > 0:
            print(f"✅ {removed_count} temporary files removed")
        else:
            print("ℹ️ No temporary files to remove")
    
    def print_statistics(self):
        """Print complete statistics"""
        if self.results is None:
            return
        
        print(f"\n{'='*70}")
        print(f"FINAL STATISTICS - SMART BUFFER ANALYSIS - {self.city_name}")
        print(f"{'='*70}")
        
        print(f"🔲 Optimized grid: {len(self.results)} cells of {self.grid_size_m}m")
        print(f"🎯 FRC_URB2D threshold: ≥{self.frc_threshold:.0%}")
        
        # Analysis boundary statistics
        if self.analysis_boundary is not None:
            boundary_area_km2 = self.analysis_boundary.geometry.area.iloc[0] / 1_000_000
            # Try to determine boundary method from the analysis boundary metadata
            boundary_method = "analysis boundary"  # Generic fallback
            print(f"🗺️ Analysis boundary: {boundary_area_km2:.1f} km² ({boundary_method})")
        
        # Building statistics
        if self.buildings is not None:
            print(f"🏢 Buildings analyzed: {len(self.buildings)} total")
        
        # Centroids statistics
        if self.unified_centroids is not None:
            building_centroids = (self.unified_centroids['feature_type'] == 'building').sum()
            impervious_centroids = (self.unified_centroids['feature_type'] == 'impervious').sum()
            print(f"📍 Centroids analyzed: {len(self.unified_centroids)} total")
            print(f"   🏢 Building centroids: {building_centroids}")
            print(f"   🏗️ Impervious centroids: {impervious_centroids}")
        
        # Road statistics
        if self.roads is not None:
            print(f"🛣️ Roads: {len(self.roads)} segments")
        
        # Impervious surface statistics
        if 'impervious_surfaces' in self.other_features:
            impervious = self.other_features['impervious_surfaces']
            print(f"🏗️ Impervious surfaces: {len(impervious)} total")
        
        print(f"\n📊 WRF PARAMETERS:")
        wrf_params = ['FRC_URB2D', 'MH_URB2D', 'BLDFR_URB2D', 'H2W', 'BW', 'Z0', 'ZD', 'Sigma']
        
        for param in wrf_params:
            if param in self.results.columns:
                values = self.results[param].dropna()  # Exclude NA values
                if len(values) > 0:
                    print(f"  {param}: μ={values.mean():.3f}, [{values.min():.3f}, {values.max():.3f}]")
                else:
                    print(f"  {param}: All NA")
        
        # Urban cells
        urban_cells = (self.results['FRC_URB2D'] >= self.frc_threshold).sum()
        urban_pct = (urban_cells / len(self.results)) * 100
        print(f"\n🏙️ Urban cells (FRC_URB2D ≥ {self.frc_threshold:.0%}): {urban_cells} ({urban_pct:.1f}%)")
        
        # NA cells
        na_cells = self.results['FRC_URB2D'].isna().sum()
        na_pct = (na_cells / len(self.results)) * 100
        print(f"🌿 NA cells (below threshold): {na_cells} ({na_pct:.1f}%)")
        
        # Efficiency statistics
        if self.unified_centroids is not None and self.analysis_boundary is not None:
            boundary_area_km2 = self.analysis_boundary.geometry.area.iloc[0] / 1_000_000
            centroid_density = len(self.unified_centroids) / boundary_area_km2
            print(f"\n📈 EFFICIENCY METRICS:")
            print(f"   Centroid density: {centroid_density:.1f} features/km²")
            print(f"   Grid efficiency: {len(self.results):.0f} cells vs traditional approach")
    
    def plot_results(self, parameter='FRC_URB2D', figsize=(15, 10)):
        """Visualize results on map with analysis boundary"""
        if self.results is None:
            print("❌ No results to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Parameter values
        data_to_plot = self.results.dropna(subset=[parameter])
        if len(data_to_plot) > 0:
            data_to_plot.plot(column=parameter, cmap='viridis', legend=True, ax=ax1, alpha=0.8)
        
        # Plot NA cells in gray
        na_cells = self.results[self.results[parameter].isna()]
        if len(na_cells) > 0:
            na_cells.plot(color='lightgray', ax=ax1, alpha=0.5, label='NA cells')
        
        # Plot analysis boundary
        if self.analysis_boundary is not None:
            self.analysis_boundary.boundary.plot(ax=ax1, color='red', linewidth=2, alpha=0.7, label='Analysis boundary')
        
        ax1.set_title(f'{parameter} - {self.city_name}\n(Urban grid {self.grid_size_m}m, Threshold ≥{self.frc_threshold:.0%})')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]') 
        ax1.axis('equal')
        ax1.legend()
        
        # Plot 2: Centroids and boundary
        if self.unified_centroids is not None:
            # Plot building centroids
            building_centroids = self.unified_centroids[self.unified_centroids['feature_type'] == 'building']
            if len(building_centroids) > 0:
                building_centroids.plot(ax=ax2, color='blue', markersize=1, alpha=0.6, label='Building centroids')
            
            # Plot impervious centroids
            impervious_centroids = self.unified_centroids[self.unified_centroids['feature_type'] == 'impervious']
            if len(impervious_centroids) > 0:
                impervious_centroids.plot(ax=ax2, color='orange', markersize=2, alpha=0.8, label='Impervious centroids')
        
        # Plot analysis boundary
        if self.analysis_boundary is not None:
            self.analysis_boundary.boundary.plot(ax=ax2, color='red', linewidth=2, label='Analysis boundary')
            self.analysis_boundary.plot(ax=ax2, color='red', alpha=0.1)
        
        # Plot grid
        self.grid.boundary.plot(ax=ax2, color='black', linewidth=0.5, alpha=0.3, label='Grid cells')
        
        ax2.set_title(f'Urban Analysis - {self.city_name}\n({len(self.unified_centroids) if self.unified_centroids is not None else 0} centroids, {len(self.results)} optimized cells)')
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.axis('equal')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
        plot_file = os.path.join(self.output_dir, f"urban_analysis_{city_clean}_{self.grid_size_m}m.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_file}")
        
        plt.show()
    
    def plot_boundary_creation_process(self, figsize=(15, 10)):
        """Plot detailed smart buffer boundary creation process"""
        if self.analysis_boundary is None:
            print("❌ No analysis boundary to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Raw centroids
        ax1 = axes[0]
        if self.unified_centroids is not None:
            building_centroids = self.unified_centroids[self.unified_centroids['feature_type'] == 'building']
            impervious_centroids = self.unified_centroids[self.unified_centroids['feature_type'] == 'impervious']
            
            if len(building_centroids) > 0:
                building_centroids.plot(ax=ax1, color='blue', markersize=1, alpha=0.7, label=f'Buildings ({len(building_centroids)})')
            if len(impervious_centroids) > 0:
                impervious_centroids.plot(ax=ax1, color='orange', markersize=2, alpha=0.8, label=f'Impervious ({len(impervious_centroids)})')
        
        ax1.set_title('Step 1: Raw Centroids')
        ax1.legend()
        ax1.set_aspect('equal')
        
        # Plot 2: Buffer process (simulate)
        ax2 = axes[1]
        if self.unified_centroids is not None:
            # Show sample of buffers
            sample_centroids = self.unified_centroids.sample(min(100, len(self.unified_centroids)))
            for idx, centroid in sample_centroids.iterrows():
                buffer = centroid.geometry.buffer(350)  # Use typical buffer distance
                gpd.GeoSeries([buffer]).plot(ax=ax2, alpha=0.3, color='lightblue', edgecolor='blue')
        
        ax2.set_title('Step 2: Buffer Creation (350m)')
        ax2.set_aspect('equal')
        
        # Plot 3: Final boundary
        ax3 = axes[2]
        if self.unified_centroids is not None:
            self.unified_centroids.plot(ax=ax3, color='gray', markersize=0.5, alpha=0.5)
        
        if self.analysis_boundary is not None:
            self.analysis_boundary.boundary.plot(ax=ax3, color='red', linewidth=3, label='Smart Buffer Boundary')
            self.analysis_boundary.plot(ax=ax3, color='red', alpha=0.2)
        
        ax3.set_title('Step 3: Final Boundary')
        ax3.legend()
        ax3.set_aspect('equal')
        
        # Plot 4: Grid overlay
        ax4 = axes[3]
        if self.analysis_boundary is not None:
            self.analysis_boundary.boundary.plot(ax=ax4, color='red', linewidth=2, label='Boundary')
            self.analysis_boundary.plot(ax=ax4, color='red', alpha=0.1)
        
        if self.grid is not None:
            self.grid.boundary.plot(ax=ax4, color='black', linewidth=0.8, alpha=0.7, label=f'Grid ({len(self.grid)} cells)')
        
        ax4.set_title('Step 4: Optimized Grid')
        ax4.legend()
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save plot
        city_clean = self.city_name.split(',')[0].replace(' ', '_').replace("'", "").lower()
        process_plot_file = os.path.join(self.output_dir, f"smart_buffer_process_{city_clean}.png")
        plt.savefig(process_plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Boundary process plot saved: {process_plot_file}")
        
        plt.show()
    
    def run_smart_buffer_workflow(self, buffer_distance_m=None, 
                                 density_similarity_threshold=None,
                                 min_area_km2=None, debug_parallel=False, use_easy_boundary=False):
        """
        Run complete SMART BUFFER workflow with adaptive parameters:
        1. Download ALL buildings (unrestricted)
        2. Download ALL impervious surfaces (unrestricted)  
        3. Extract unified centroids
        4. Create analysis boundary using SMART BUFFER with adaptive density-based selection
           OR EASY BOUNDARY using OSMnx administrative boundary
        5. Create optimized grid within boundary
        6. Download roads
        7. Calculate parameters with threshold
        
        Args:
            buffer_distance_m: Buffer distance (None = adaptive: larger for smaller cities)
            density_similarity_threshold: Density similarity (None = dynamic calculation)
            min_area_km2: Minimum area (None = adaptive)
            debug_parallel: Enable parallel debugging
            use_easy_boundary: Use OSMnx administrative boundary instead of smart buffer
        """
        boundary_method = "EASY BOUNDARY (OSMnx)" if use_easy_boundary else "SMART BUFFER (adaptive)"
        
        print(f"\n🚀 STARTING URBAN ANALYSIS WORKFLOW")
        print(f"City: {self.city_name}")
        print(f"Grid: {self.grid_size_m}m")
        print(f"FRC_URB2D threshold: ≥{self.frc_threshold:.0%}")
        print(f"Boundary method: {boundary_method}")
        print(f"Output: {self.output_dir}")
        
        steps = [
            ("Download ALL buildings (unrestricted)", self.step1_download_buildings_unrestricted),
            ("Download ALL impervious surfaces (unrestricted)", self.step2_download_impervious_surfaces_unrestricted),
            ("Extract unified centroids", self.step3_extract_unified_centroids),
            ("Create analysis boundary", lambda: self.step4_create_analysis_boundary_smart_buffer(
                buffer_distance_m, density_similarity_threshold, min_area_km2, use_easy_boundary)),
            ("Create optimized grid", self.step5_create_optimized_grid),
            ("Download road network", self.step6_download_roads),
            ("Process unified features (overlap resolution)", self.step6_5_process_unified_features),
            ("Calculate parameters with threshold", lambda: self.step7_calculate_parameters_per_cell(debug_parallel=debug_parallel))
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*60}")
            success = step_func()
            if not success:
                print(f"❌ FAILED: {step_name}")
                return False
            print(f"✅ COMPLETED: {step_name}")
        
        # Save results and statistics
        self.save_results()
        self.print_statistics()
        
        print(f"\n🎉 URBAN ANALYSIS WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"🔍 Boundary method used: {boundary_method}")
        
        return True


# === CONVENIENCE FUNCTION ===
def calculate_urban_parameters_smart_buffer(city_name, grid_size_m=1000, output_dir="output", 
                                           n_cores=None, debug_mode=False, frc_threshold=0.20,
                                           buffer_distance_m=None, density_similarity_threshold=None,
                                           min_area_km2=None, use_easy_boundary=False):
    """
    Smart buffer urban parameter calculation for WRF with adaptive parameters
    
    Args:
        city_name (str): City name (e.g. "Rome, Italy")
        grid_size_m (int): Grid size in meters (default: 1000)
        output_dir (str): Output directory
        n_cores (int): Number of cores for parallel computation
        debug_mode (bool): Enable detailed debugging
        frc_threshold (float): FRC_URB2D threshold (0.20 = 20%)
        buffer_distance_m (int): Buffer distance (None = adaptive: larger for smaller cities)
        density_similarity_threshold (float): Density similarity (None = dynamic calculation)
        min_area_km2 (float): Minimum area for component (None = adaptive)
        use_easy_boundary (bool): Use OSMnx administrative boundary instead of smart buffer
    
    Returns:
        UrbanMorphometricWorkflow: Workflow object with results, or None if error
    """
    boundary_method = "Easy boundary (OSMnx)" if use_easy_boundary else "Smart buffer with adaptive parameters"
    
    print(f"🏙️ URBAN PARAMETERS FOR WRF")
    print(f"=" * 55)
    print(f"City: {city_name}")
    print(f"Approach: {boundary_method}")
    print(f"Grid: {grid_size_m}m x {grid_size_m}m") 
    print(f"FRC_URB2D threshold: ≥{frc_threshold:.0%}")
    print(f"Output: {output_dir}")
    
    workflow = UrbanMorphometricWorkflow(
        city_name=city_name,
        grid_size_m=grid_size_m, 
        output_dir=output_dir,
        n_cores=n_cores,
        debug_mode=debug_mode,
        frc_threshold=frc_threshold
    )
    
    success = workflow.run_smart_buffer_workflow(
        buffer_distance_m=buffer_distance_m,
        density_similarity_threshold=density_similarity_threshold,
        min_area_km2=min_area_km2,
        use_easy_boundary=use_easy_boundary
    )
    
    if success:
        return workflow
    else:
        return None


# === CLI MAIN FUNCTION ===
def main():
    """Entry point for smart buffer urbanindex CLI command"""
    
    parser = argparse.ArgumentParser(
        description="🏙️ UrbanIndex - Urban Morphometric Parameters for WRF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  urbanindex --city "Rome, Italy"                              # Smart buffer (adaptive)
  urbanindex --city "Milan, Italy" --easy-boundary             # Easy boundary (OSMnx)
  urbanindex --city "Naples, Italy" --buffer 400               # Custom buffer
  urbanindex --city "Florence, Italy" --threshold 0.15         # Custom FRC threshold
        """
    )
    
    # Main arguments
    parser.add_argument("--city", required=True, help='City name (e.g. "Rome, Italy")')
    parser.add_argument("--grid", type=int, default=1000, help="Grid size in meters (default: 1000)")
    parser.add_argument("--output", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--cores", type=int, default=None, help="Number of cores for parallel computation")
    parser.add_argument("--threshold", type=float, default=0.20, help="FRC_URB2D threshold (default: 0.20 = 20%%)")
    
    # Boundary method selection
    parser.add_argument("--easy-boundary", action="store_true", 
                       help="Use OSMnx administrative boundary instead of smart buffer (faster but may include rural areas)")
    
    # Smart buffer specific arguments (ignored if --easy-boundary is used)
    parser.add_argument("--buffer", type=int, default=None, help="Buffer distance in meters (default: adaptive - larger for smaller cities)")
    parser.add_argument("--density-similarity", type=float, default=None, help="Density similarity threshold (default: dynamic calculation)")
    parser.add_argument("--min-area", type=float, default=None, help="Minimum component area in km² (default: adaptive)")
    
    # Advanced options
    parser.add_argument("--force-download", action="store_true", help="Force re-download data")
    parser.add_argument("--clean-temp", action="store_true", help="Clean temporary files after completion")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate visualization plots")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--version", action="version", version="UrbanIndex 3.1.0 - Smart Buffer Analysis")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.01 <= args.threshold <= 1.0:
        print("❌ Error: threshold must be between 0.01 and 1.0")
        return 1
    
    if args.buffer is not None and not 100 <= args.buffer <= 1000:
        print("❌ Error: buffer must be between 100 and 1000 meters")
        return 1
        
    if args.density_similarity is not None and not 0.05 <= args.density_similarity <= 0.50:
        print("❌ Error: density-similarity must be between 0.05 and 0.50")
        return 1
        
    if args.min_area is not None and not 0.1 <= args.min_area <= 10.0:
        print("❌ Error: min-area must be between 0.1 and 10.0 km²")
        return 1
    
    # Warning for unused arguments with easy boundary
    if args.easy_boundary and (args.buffer or args.density_similarity or args.min_area):
        print("⚠️ Warning: --buffer, --density-similarity, and --min-area are ignored when using --easy-boundary")
    
    # Configure output directory
    output_dir = os.path.abspath(args.output)
    
    # Determine boundary method for display
    boundary_method = "EASY BOUNDARY (OSMnx)" if args.easy_boundary else "SMART BUFFER (adaptive)"
    
    # Header
    print("🏙️" + "="*70)
    print("     URBANINDEX - Urban Morphometric Parameters")
    print("="*73)
    print(f"📍 City: {args.city}")
    print(f"🔲 Grid: {args.grid}m x {args.grid}m")
    print(f"🎯 FRC_URB2D threshold: ≥{args.threshold:.0%}")
    print(f"🗺️ Boundary method: {boundary_method}")
    
    if not args.easy_boundary:
        print(f"🔵 Buffer distance: {'ADAPTIVE (larger for smaller cities)' if args.buffer is None else f'{args.buffer}m'}")
        print(f"📊 Density similarity: {'DYNAMIC (calculated from data)' if args.density_similarity is None else f'±{args.density_similarity*100:.0f}%'}")
        print(f"📐 Min component area: {'ADAPTIVE' if args.min_area is None else f'{args.min_area} km²'}")
    else:
        print(f"🏛️ Using official administrative boundary from OSMnx")
    
    print(f"📁 Output: {output_dir}")
    print("="*73)
    
    try:
        # Initialize workflow
        workflow = UrbanMorphometricWorkflow(
            city_name=args.city,
            grid_size_m=args.grid, 
            output_dir=output_dir,
            n_cores=args.cores,
            debug_mode=args.debug,
            frc_threshold=args.threshold
        )
        
        # Force re-download if requested
        if args.force_download:
            workflow._force_redownload = True
        
        # Run urban analysis workflow
        success = workflow.run_smart_buffer_workflow(
            buffer_distance_m=args.buffer,
            density_similarity_threshold=args.density_similarity,
            min_area_km2=args.min_area,
            use_easy_boundary=args.easy_boundary
        )
        
        if success:
            # Save results
            wrf_file, detailed_file, csv_file = workflow.save_results()
            
            # Statistics
            if args.verbose:
                workflow.print_statistics()
            
            # Plots (optional)
            if not args.no_plot:
                try:
                    workflow.plot_results('FRC_URB2D')
                    workflow.plot_boundary_creation_process()  # New plot for process
                except Exception as e:
                    print(f"⚠️ Cannot generate plots: {e}")
            
            # Clean temporary files if requested
            if args.clean_temp:
                workflow.clean_temp_files()
            
            # Final success message
            boundary_method_used = "easy boundary (OSMnx)" if args.easy_boundary else "smart buffer"
            print(f"\n🎉 SUCCESS! Urban parameters calculated for {args.city}")
            print(f"📄 WRF CSV file: {os.path.basename(csv_file)}")
            print(f"📄 Analysis boundary created using {boundary_method_used} method")
            
            return 0
            
        else:
            print(f"\n❌ ERROR: Smart buffer workflow failed for {args.city}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Operation interrupted by user")
        return 1
    except Exception as e:
        error_msg = str(e).lower()
        if 'sqlite' in error_msg or 'gpkg' in error_msg or 'database' in error_msg:
            print(f"\n❌ DATABASE ERROR: {e}")
            print("💡 Suggested solutions:")
            print("   - Free up disk space")
            print("   - Try --output pointing to different drive")
            print("   - Use --buffer 250 for smaller analysis area")
        else:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("🏙️ UrbanIndex - Urban Analysis")
        print("=" * 50)
        print("\nQuick start:")
        print('  python urbanindex_smart_buffer.py --city "Rome, Italy"')
        print('  python urbanindex_smart_buffer.py --city "Milan, Italy" --easy-boundary')
        print('  python urbanindex_smart_buffer.py --city "Naples, Italy" --buffer 400')
        
        # Run example
        print("\n" + "="*50)
        print("Running example: Rome, Italy with adaptive smart buffer...")
        example_workflow = calculate_urban_parameters_smart_buffer(
            city_name="Rome, Italy",
            grid_size_m=1000,
            frc_threshold=0.20,
            use_easy_boundary=False  # Try smart buffer first
            # All other parameters = None → ADAPTIVE
        )
        
        if example_workflow:
            print("✅ Example completed successfully!")
        else:
            print("❌ Example failed")
        
        sys.exit(0)
    else:
        sys.exit(main())