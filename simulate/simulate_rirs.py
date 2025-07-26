#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RIR data generation script with functional design and incremental metadata writing
"""

import os
import glob
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger

from audiblelight import utils
from audiblelight.worldstate import WorldState


@dataclass
class SimulationConfig:
    """Configuration parameters for RIR simulation"""
    n_sources: int = 1
    n_groups: int = 10
    group_size: int = 2
    mic_type: str = "ambeovr"
    seed: int = utils.SEED
    
    # Azimuth and elevation statistics
    azimuth_mean: float = 0.0
    azimuth_std: float = 125.0
    azimuth_range: Tuple[float, float] = (-180.0, 180.0)
    
    elevation_mean: float = 0.0
    elevation_std: float = 15.0
    elevation_range: Tuple[float, float] = (-79.0, 79.0)
    
    # Distance range for source placement
    distance_min: float = 0.8
    distance_max: float = 3.0

    max_reries: int = 5  # Maximum retries for RIR simulation


def get_gibson_meshes(dataset_dir: str) -> List[str]:
    """
    Get all Gibson mesh files from the dataset directory
    
    Args:
        dataset_dir: Path to the Gibson dataset directory
        
    Returns:
        List of full paths to Gibson mesh files
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    mesh_paths = glob.glob('*.glb', root_dir=dataset_dir)
    gibson_files = [
        os.path.join(dataset_dir, mesh_file) 
        for mesh_file in mesh_paths 
        if os.path.isfile(os.path.join(dataset_dir, mesh_file))
    ]
    
    if not gibson_files:
        raise ValueError(f"No Gibson mesh files found in {dataset_dir}")
    
    logger.info(f"Found {len(gibson_files)} Gibson mesh files")
    return gibson_files


def sample_spherical_coordinates(config: SimulationConfig) -> Tuple[float, float, float]:
    """
    Sample spherical coordinates based on the provided statistics
    
    Args:
        config: Simulation configuration containing statistical parameters
        
    Returns:
        Tuple of (azimuth, elevation, distance) in degrees and meters
    """
    # Sample azimuth with clipping to valid range
    azimuth = np.random.normal(config.azimuth_mean, config.azimuth_std)
    azimuth = np.clip(azimuth, config.azimuth_range[0], config.azimuth_range[1])
    
    # Sample elevation with clipping to valid range
    elevation = np.random.normal(config.elevation_mean, config.elevation_std)
    elevation = np.clip(elevation, config.elevation_range[0], config.elevation_range[1])
    
    # Sample distance uniformly within specified range
    distance = np.random.uniform(config.distance_min, config.distance_max)
    
    return float(azimuth), float(elevation), float(distance)


def create_world_state_and_simulate(mesh_path: str, config: SimulationConfig, coordinates: tuple) -> Dict[str, Any]:
    """
    Create a WorldState, place microphone and emitter, and simulate RIRs
    
    Args:
        mesh_path: Path to the mesh file
        config: Simulation configuration
        coordinates: Tuple of (azimuth, elevation, distance) for emitter placement
        
    Returns:
        Dictionary containing simulation results and metadata
    """
    space_name = os.path.basename(mesh_path).split('.')[0]
    logger.info(f"Processing {space_name}")

    azimuth, elevation, distance = coordinates
    
    try:
        # Create WorldState
        world_state = WorldState(mesh_path)
        
        for attempt in range(config.max_reries):
            try:
                # Add microphone and emitter with spherical relationship
                world_state.add_microphone_emitter_spherical(
                    azimuth=azimuth,
                    elevation=elevation, 
                    distance=distance,
                    microphone_type=config.mic_type,
                    mic_alias="ir_mic",
                    emitter_alias="source"
                )
                logger.info(f"Successfully added microphone and emitter for {space_name}")
                break  # Exit loop if successful
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{config.max_reries} failed for {space_name}: {str(e)}")
                if attempt == config.max_reries - 1:
                    raise RuntimeError(f"Failed to add microphone/emitter after {config.max_reries} attempts")
        # Simulate RIRs
        world_state.simulate()
        
        # Get ray efficiency
        ray_efficiency = world_state.ctx.get_indirect_ray_efficiency()
        
        # Extract IRs and positions
        irs = world_state.microphones["ir_mic"].irs
        mic_center = world_state.microphones["ir_mic"].coordinates_center
        mic_capsules = world_state.microphones["ir_mic"].coordinates_absolute
        source_position = world_state.get_emitter("source", 0).coordinates_absolute
        
        # Sanity check
        n_caps, n_sources, n_samples = irs.shape
        assert n_caps == 4, f"Expected 4 capsules, got {n_caps}"
        assert n_sources == config.n_sources, f"Expected {config.n_sources} sources, got {n_sources}"
        
        # Check if source is visible (direct path exists)
        source_visible = world_state.path_exists_between_points(mic_center, source_position)
        
        return {
            "space_name": space_name,
            "mesh_path": mesh_path,
            "mic_center_position": mic_center,
            "mic_capsules_position": mic_capsules,
            "source_position": source_position,
            "mesh_bounds": world_state.mesh.bounds,
            "irs": irs,
            "ray_efficiency": ray_efficiency,
            "source_visible": source_visible,
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance
        }
        
    except Exception as e:
        logger.error(f"Failed to process {space_name}: {str(e)}")
        return None


def initialize_metadata_file(config: SimulationConfig, output_dir: str) -> str:
    """
    Initialize the metadata file with header information
    
    Args:
        config: Simulation configuration
        output_dir: Output directory path
        
    Returns:
        Path to the metadata file
    """
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    # Check if metadata file already exists
    if os.path.exists(metadata_path):
        logger.info(f"Metadata file already exists: {metadata_path}")
        logger.info("Appending to existing metadata file")
        return metadata_path
    
    # Create initial metadata structure
    initial_metadata = {
        "seed": config.seed,
        "dataset": "Gibson Habitat",
        "version": "1.0",
        "config": {
            "n_sources": config.n_sources,
            "n_groups": config.n_groups,
            "group_size": config.group_size,
            "mic_type": config.mic_type,
            "azimuth_stats": {
                "mean": config.azimuth_mean,
                "std": config.azimuth_std,
                "range": config.azimuth_range
            },
            "elevation_stats": {
                "mean": config.elevation_mean,
                "std": config.elevation_std,
                "range": config.elevation_range
            },
            "distance_range": [config.distance_min, config.distance_max]
        },
        "groups": {},
        "summary": {
            "total_groups": 0,
            "total_simulations": 0,
            "completed_groups": []
        }
    }
    
    # Write initial metadata
    with open(metadata_path, 'w') as f:
        json.dump(initial_metadata, f, indent=2)
    
    logger.info(f"Initialized metadata file: {metadata_path}")
    return metadata_path


def initialize_group_in_metadata(metadata_path: str, group_idx: int, coordinates: Tuple[float, float, float]) -> None:
    """
    Initialize a new group in the metadata file
    
    Args:
        metadata_path: Path to the metadata file
        group_idx: Group index
        coordinates: Spherical coordinates used for this group
    """
    try:
        # Read existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        azimuth, elevation, distance = coordinates
        
        # Initialize group if it doesn't exist
        group_key = f"group_{group_idx}"
        if group_key not in metadata["groups"]:
            metadata["groups"][group_key] = {
                "group_id": group_idx,
                "azimuth": azimuth,
                "elevation": elevation,
                "distance": distance,
                "rooms": [],
                "n_completed": 0,
                "timestamp_started": None,
                "timestamp_completed": None
            }
            
            # Update summary
            metadata["summary"]["total_groups"] += 1
            
            # Write back to file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Initialized group {group_idx} in metadata")
        
    except Exception as e:
        logger.error(f"Failed to initialize group in metadata file: {str(e)}")
        raise


def append_to_group_metadata(metadata_path: str, group_idx: int, new_entry: Dict[str, Any]) -> None:
    """
    Append a new entry to a specific group in the metadata file
    
    Args:
        metadata_path: Path to the metadata file
        group_idx: Group index
        new_entry: New metadata entry to append
    """
    try:
        # Read existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        group_key = f"group_{group_idx}"
        
        # Ensure group exists
        if group_key not in metadata["groups"]:
            logger.error(f"Group {group_idx} not found in metadata")
            raise ValueError(f"Group {group_idx} not initialized")
        
        # Append new entry to group
        metadata["groups"][group_key]["rooms"].append(new_entry)
        metadata["groups"][group_key]["n_completed"] += 1
        
        # Update global summary
        metadata["summary"]["total_simulations"] += 1
        
        # Write back to file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.debug(f"Appended entry to group {group_idx} (total in group: {metadata['groups'][group_key]['n_completed']})")
        
    except Exception as e:
        logger.error(f"Failed to append to group metadata: {str(e)}")
        raise


def complete_group_in_metadata(metadata_path: str, group_idx: int) -> None:
    """
    Mark a group as completed in the metadata
    
    Args:
        metadata_path: Path to the metadata file
        group_idx: Group index
    """
    try:
        # Read existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        group_key = f"group_{group_idx}"
        
        if group_key in metadata["groups"]:
            # Mark group as completed
            from datetime import datetime
            metadata["groups"][group_key]["timestamp_completed"] = datetime.now().isoformat()
            
            # Add to completed groups list
            if group_idx not in metadata["summary"]["completed_groups"]:
                metadata["summary"]["completed_groups"].append(group_idx)
            
            # Write back to file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.debug(f"Marked group {group_idx} as completed")
        
    except Exception as e:
        logger.error(f"Failed to complete group in metadata: {str(e)}")
        raise


def save_simulation_data(sim_data: Dict[str, Any], output_dir: str, 
                        group_idx: int, room_idx: int, metadata_path: str) -> None:
    """
    Save simulation data to file and append metadata entry to the appropriate group
    
    Args:
        sim_data: Simulation results dictionary
        output_dir: Output directory path
        group_idx: Group index
        room_idx: Room index within group
        metadata_path: Path to the metadata file
    """
    # Create filename
    fname = f"{sim_data['space_name']}/{group_idx}_{str(room_idx).zfill(2)}.npy"
    filepath = os.path.join(output_dir, fname)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save IRs data
    np.save(filepath, sim_data["irs"])
    
    # Format positions as comma-separated strings
    sensor_pos_str = ",".join(f"{x:.4f}" for x in sim_data["mic_center_position"])
    source_pos_str = ",".join(f"{x:.4f}" for x in sim_data["source_position"])
    
    # Create metadata entry
    metadata_entry = {
        "fname": fname,
        "sensor_position": sensor_pos_str,
        "source_position": source_pos_str,
        "room_id": room_idx,
        "space_name": sim_data['space_name'],
        "fixed_receiver": True,
        "sourceIsVisible": sim_data["source_visible"],
        "RayEfficiency": sim_data["ray_efficiency"],
        "azimuth": sim_data["azimuth"],
        "elevation": sim_data["elevation"],
        "distance": sim_data["distance"]
    }
    
    # Append to the appropriate group in metadata file immediately
    append_to_group_metadata(metadata_path, group_idx, metadata_entry)


def generate_rir_dataset(config: SimulationConfig, dataset_dir: str, output_dir: str) -> None:
    """
    Main function to generate RIR dataset with incremental metadata writing
    
    Args:
        config: Simulation configuration
        dataset_dir: Path to Gibson dataset directory
        output_dir: Path to output directory
    """
    # Set seed for reproducibility
    utils.seed_everything(config.seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metadata file
    metadata_path = initialize_metadata_file(config, output_dir)
    
    # Get Gibson mesh files
    gibson_files = get_gibson_meshes(dataset_dir)
    
    # Track statistics
    total_processed = 0
    total_failed = 0
    
    # Process each group
    for group_idx in range(config.n_groups):
        logger.info(f"Processing group {group_idx + 1}/{config.n_groups}")
        
        # Randomly select rooms for this group
        selected_indices = np.random.choice(
            len(gibson_files), 
            size=config.group_size, 
            replace=False
        )
        selected_rooms = [gibson_files[idx] for idx in selected_indices]

        # Sample spherical coordinates for emitter placement
        coordinates = sample_spherical_coordinates(config)
        
        # Initialize group in metadata
        initialize_group_in_metadata(metadata_path, group_idx, coordinates)
        
        # Set group start timestamp
        from datetime import datetime
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata["groups"][f"group_{group_idx}"]["timestamp_started"] = datetime.now().isoformat()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to set start timestamp for group {group_idx}: {str(e)}")
        
        group_processed = 0
        group_failed = 0
        
        # Process each room in the group
        for room_idx, mesh_path in enumerate(selected_rooms):
            try:
                # Simulate and get results
                sim_data = create_world_state_and_simulate(mesh_path, config, coordinates)
                
                if sim_data is not None:
                    # Save data and append to metadata immediately
                    save_simulation_data(
                        sim_data, output_dir, group_idx, room_idx, metadata_path
                    )
                    total_processed += 1
                    group_processed += 1
                    
                    logger.info(f"Successfully processed {sim_data['space_name']} "
                              f"(Group {group_idx}, Room {room_idx}) - "
                              f"Group: {group_processed}/{config.group_size}, "
                              f"Total: {total_processed}")
                else:
                    total_failed += 1
                    group_failed += 1
                    logger.warning(f"Skipped failed simulation for group {group_idx}, "
                                 f"room {room_idx} - Group failed: {group_failed}, "
                                 f"Total failed: {total_failed}")
                    
            except Exception as e:
                total_failed += 1
                group_failed += 1
                logger.error(f"Error processing group {group_idx}, room {room_idx}: {str(e)} "
                           f"- Group failed: {group_failed}, Total failed: {total_failed}")
                continue
        
        # Mark group as completed
        complete_group_in_metadata(metadata_path, group_idx)
        
        logger.info(f"Completed group {group_idx}: {group_processed} successful, "
                   f"{group_failed} failed simulations")
        
        # Log progress every 10 groups
        if (group_idx + 1) % 10 == 0:
            logger.info(f"Progress: {group_idx + 1}/{config.n_groups} groups completed. "
                       f"Total processed: {total_processed}, Total failed: {total_failed}")
    
    logger.info(f"Dataset generation complete!")
    logger.info(f"Successfully processed: {total_processed} simulations")
    logger.info(f"Failed simulations: {total_failed}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Metadata saved to: {metadata_path}")


def main():
    """Main execution function"""
    # Configuration
    config = SimulationConfig(
        n_sources=1,
        n_groups=1000,
        group_size=12,
        mic_type="ambeovr",
        seed=43,  # Set a specific seed for reproducibility
    )
    
    # Paths
    project_root = utils.get_project_root()
    dataset_dir = os.path.join(project_root, "/scratch/ssd1/matterport3d/gibson_habitat/gibson")
    output_dir = os.path.join(project_root, "/scratch/ssd1/DINOSAUD_Dataset")
    
    # Generate dataset
    generate_rir_dataset(config, dataset_dir, output_dir)


if __name__ == "__main__":
    main()