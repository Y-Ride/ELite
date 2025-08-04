import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
from scipy.spatial import KDTree

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


class MapRemover:
    def __init__(
        self, 
        config_path: str
    ):
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]        
        os.makedirs(p_settings["output_dir"], exist_ok=True)
        
        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1

        self.session_loader : Session = None
        self.session_map : SessionMap = None


    def load(self, new_session : Session = None):
        p_settings = self.params["settings"]

        if new_session is None:
            self.session_loader = Session(p_settings["scans_dir"], p_settings["poses_file"])
        else:
            self.session_loader = new_session
        
        logger.info(f"Loaded new session")


    def run(self):
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]
        checkpoint_dir = os.path.join(p_settings["output_dir"], "checkpoint_ephemerality")

        if os.path.exists(checkpoint_dir):
            res = input("Checkpoint directory found. Do you want to use the saved checkpoint? [y]/n ")
            if res == 'n':
                use_checkpoint = False
            else:
                use_checkpoint = True
        else:
            use_checkpoint = False


        if use_checkpoint:
            logger.info("Loading checkpoint...")

            anchor_points = o3d.io.read_point_cloud(os.path.join(checkpoint_dir, "anchor_points.pcd"))
            anchor_eph_l = np.load(os.path.join(checkpoint_dir, "anchor_eph_l.npy"))
            session_map = o3d.io.read_point_cloud(os.path.join(checkpoint_dir, "session_map.pcd"))
            anchor_kdtree = KDTree(np.asarray(anchor_points.points))

            logger.info("Checkpoint loaded.")
        else:
        
            logger.info(f"Starting dynamic object removal")

            logger.info(f"Aggregating {len(self.session_loader)} scans to create session map...")
            # 1) Aggregate scans to create session map
            # session_map = self.session_loader[0:len(self.session_loader)].downsample(0.01).get()

            ## Ben added
            batch_size = 100  # Adjust based on RAM
            partial_clouds = []
            total_batches = (len(self.session_loader) // batch_size) +1
            for i in range(0, len(self.session_loader), batch_size):
                logger.debug(f"Processing batch {(i//batch_size)+1} of {total_batches}")
                batch = self.session_loader[i:i + batch_size]  # Returns PointCloud
                downsampled = batch.downsample(0.01).get()     # Apply downsampling to this batch
                partial_clouds.append(downsampled)

            # Merge all downsampled batches
            logger.debug(f"Combining batch point clouds...")
            combined = o3d.geometry.PointCloud()

            for pc in partial_clouds:
                combined += pc

            session_map = combined  # Now this is the final downsampled point cloud
            ## End of added

            eph_l = np.zeros(len(session_map.points))
            logger.info(f"Initialized session map")

            # 2) Select anchor points for local ephemerality update
            anchor_points = session_map.voxel_down_sample(p_dor["anchor_voxel_size"])
            anchor_eph_l = np.ones(len(anchor_points.points)) * 0.5 # initial value
            anchor_kdtree = KDTree(np.asarray(anchor_points.points))

            logger.info(f"Updating anchor local ephemerality")
            for i in trange(0, len(self.session_loader), p_dor["stride"], desc="Updating \u03B5_l", ncols=100):

                logger.debug(f"Processing scan {i + 1}/{len(self.session_loader)}")
                scan = np.asarray(self.session_loader[i].get().points)
                pose = self.session_loader.get_pose(i)[:3, 3]
                
                # occupied space update
                dists, inds = anchor_kdtree.query(scan, k=p_dor["num_k"])
                for j in range(len(dists)):
                    dist = dists[j]
                    eph_l_prev = anchor_eph_l[inds[j]]
                    update_rate = np.minimum(self.alpha * (1 - np.exp(-1 * dist**2 / self.std_dev_o)) + self.beta, self.alpha) # Eq. 5 
                    eph_l_new = eph_l_prev * update_rate / (
                        eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                    )
                    anchor_eph_l[inds[j]] = eph_l_new

                # free space update
                shifted_scan = scan - pose # local coordinates
                sample_ratios = np.linspace(p_dor["min_ratio"], p_dor["max_ratio"], p_dor["num_samples"])
                free_space_samples = pose + shifted_scan[:, np.newaxis, :] * sample_ratios.T[np.newaxis, :, np.newaxis]
                free_space_samples = free_space_samples.reshape(-1, 3)
                free_space_samples_o3d = o3d.geometry.PointCloud()
                free_space_samples_o3d.points = o3d.utility.Vector3dVector(free_space_samples)
                free_space_samples_o3d = free_space_samples_o3d.voxel_down_sample(voxel_size=0.1)
                free_space_samples = np.asarray(free_space_samples_o3d.points)
                dists, inds = anchor_kdtree.query(free_space_samples, k=p_dor["num_k"])
                for j in range(len(dists)):
                    dist = dists[j]
                    eph_l_prev = anchor_eph_l[inds[j]]
                    update_rate = np.maximum(self.alpha * (1 + np.exp(-1 * dist**2 / self.std_dev_f)) - self.beta, self.alpha) # Eq. 5
                    eph_l_new = eph_l_prev * update_rate / (
                        eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                    )
                    anchor_eph_l[inds[j]] = eph_l_new

            # Make a checkpoint so that I can resume later
            checkpoint_dir = os.path.join(p_settings["output_dir"], "checkpoint_ephemerality")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save anchor points (Open3D format)
            o3d.io.write_point_cloud(os.path.join(checkpoint_dir, "anchor_points.pcd"), anchor_points)

            # Save ephemerality values
            np.save(os.path.join(checkpoint_dir, "anchor_eph_l.npy"), anchor_eph_l)

            # Save session map
            o3d.io.write_point_cloud(os.path.join(checkpoint_dir, "session_map.pcd"), session_map)

            logger.info(f"Checkpoint saved successfully to {checkpoint_dir} before Step 3.")

        # 3) Propagate anchor local ephemerality to session map
        logger.info(f"Propagate anchor local ephemerality to session map")
        # distances, indices = anchor_kdtree.query(np.asarray(session_map.points), k=p_dor["num_k"])
        # distances = np.maximum(distances, 1e-6) # avoid division by zero
        # weights = 1 / (distances**2)
        # weights /= np.sum(weights, axis=1, keepdims=True)
        # eph_l = np.sum(weights * anchor_eph_l[indices], axis=1)
        # eph_l = np.clip(eph_l, 0, 1) # redundant, but for safety
        batch_size = 1000000  # or smaller based on your RAM
        num_points = len(session_map.points)
        eph_l = np.zeros(num_points)
        total_batches = (num_points // batch_size) + 1

        for start in trange(0, num_points, batch_size):
            logger.debug(f"Processing anchors {start}/{num_points}")
            end = min(start + batch_size, num_points)
            batch_points = np.asarray(session_map.points[start:end])
            distances, indices = anchor_kdtree.query(batch_points, k=p_dor["num_k"])

            distances = np.maximum(distances, 1e-6)
            weights = 1 / (distances**2)
            weights /= np.sum(weights, axis=1, keepdims=True)
            eph_l[start:end] = np.sum(weights * anchor_eph_l[indices], axis=1)

        # 4) Remove dynamic objects to create cleaned session map
        logger.info(f"Remove dynamic objects to create cleaned session map")
        static_points = session_map.select_by_index(np.where(eph_l <= p_dor["dynamic_threshold"])[0])
        static_eph_l = eph_l[eph_l <= p_dor["dynamic_threshold"]]
        static_points.paint_uniform_color([0.5, 0.5, 0.5])
        dynamic_points = session_map.select_by_index(np.where(eph_l > p_dor["dynamic_threshold"])[0])
        dynamic_points.paint_uniform_color([1, 0, 0])
                  
        if p_dor["save_static_dynamic_map"]:
            logger.info(f"Saving static/dynamic points...")
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_points)  
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_points)
            logger.info(f"Static/dynamic points saved.")
        if p_dor["viz_static_dynamic_map"]:
            total_points = static_points + dynamic_points
            o3d.visualization.draw_geometries([total_points])

        logger.info(f"Cleaning session map...")
        cleaned_session_map = SessionMap(
            np.asarray(static_points.points), static_eph_l
        )
        self.session_map = cleaned_session_map

        if p_dor["save_cleaned_session_map"]:
            logger.info(f"Saving cleaned session map...")
            cleaned_session_map.save(p_settings["output_dir"], is_global=False)
            logger.info(f"Cleaned session map saved.") 
        if p_dor["viz_cleaned_session_map"]:
            logger.info(f"Visualizaing Cleaned Session Map. Press 'q' in Open3d window to continue")
            cleaned_session_map.visualize()

        return cleaned_session_map


    def get(self):
        return self.session_map
        

# Example usage
if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    # Load session using the config file or from an alingment module
    remover.load()
    # Run the dynamic object removal
    remover.run()
    # Get the cleaned session map
    cleaned_session_map = remover.get()