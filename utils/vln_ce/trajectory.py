from pathlib import Path
from typing import List, Callable
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pyparsing import Iterable
from utils import Traj, Trajectories

from scipy.spatial.transform import Rotation
from utils.coordinate import homogeneous_inv

class VLN_CE_Traj(Traj):
    # [4, 4]
    T_b_c = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # T_c_b == homogeneous_inv(T_b_c)
    # [4, 4]
    T_c_b = np.array([[ 0., -1.,  0., -0.],
       [ 0.,  0., -1., -0.],
       [ 1.,  0.,  0., -0.],
       [ 0.,  0.,  0.,  1.]
    ], dtype=np.float32)

    goal_keys = [
        "relative_goal_frame_id.125cm_30deg",
        "relative_goal_frame_id.125cm_45deg",
        "relative_goal_frame_id.60cm_15deg",
        "relative_goal_frame_id.60cm_30deg",
    ]

    REASON_COL = "125cm_0deg_reason"

    @property
    def metadata(self) -> dict:
        return {
            "T_b_c": self.T_b_c,
        }

    def __init__(self, parquet_path: Path, images: List[Path], task: str, task_idx: int):
        self.parquet_path = parquet_path
        self.images = images
        self.task = task
        self.task_idx = task_idx

        # Read parquet
        self.df = pd.read_parquet(self.parquet_path)

        assert len(self.df) == len(self.images), \
            f"Length mismatch between parquet {len(self.df)} and images {len(self.images)} in {self.parquet_path}"

        self.length = len(self.df)

    def _process_traj(self):
        # Process Poses
        # Ensure it's a stacked numpy array [N, 4, 4]
        # df["pose.125cm_0deg"] usually contains arrays/lists
        pose_col = self.df["pose.125cm_0deg"].to_numpy()
        T_w_c = np.array([np.stack(p) for p in pose_col])
        
        # Calculate Body Poses
        T_w_b = T_w_c @ self.T_c_b
        
        # First frame as body frame origin for state
        self.T_w_body_0 = T_w_b[0]
        T_body_w = homogeneous_inv(self.T_w_body_0) # body means first frame's body coordinate
        
        # Transform all frames to be relative to the first frame (body frame)
        # [4, 4] @ [N, 4, 4] -> [N, 4, 4]
        T_body_b = np.einsum('ij,njk->nik', T_body_w, T_w_b)
        self.poses_body = self.get_poses(T_body_b) # observation.state
        
        # Calculate Action Deltas (first 4 values)
        # Delta is relative pose from current to next frame
        T_body_b_current = T_body_b[:-1]
        T_b_current_body = homogeneous_inv(T_body_b_current)
        T_body_b_next = T_body_b[1:]
        
        # [N-1, 4, 4]
        T_current_next = T_b_current_body @ T_body_b_next
        deltas = self.get_poses(T_current_next)
        
        # Pad last frame with identity (no movement)
        last_delta = np.zeros((1, 4), dtype=np.float32) 
        self.action_deltas = np.concatenate([deltas, last_delta], axis=0) # [N, 4]
        
        # Calculate Goal Actions (last 4 values)
        # Collect arrays for all keys that exist in the dataframe
        available_goal_arrays = []
        for key in self.goal_keys:
            if key in self.df.columns:
                available_goal_arrays.append(self.df[key].to_numpy())
        
        if not available_goal_arrays:
            raise KeyError(f"None of the goal keys {self.goal_keys} found in dataframe columns: {self.df.columns}")
        
        # For each row, use the first key (in order) whose value is not -1;
        # fall back to -1 if all available keys have -1 for that row.
        goal_frame_idx = np.full(self.length, -1, dtype=available_goal_arrays[0].dtype)
        for arr in reversed(available_goal_arrays):
            mask = arr != -1
            goal_frame_idx[mask] = arr[mask]

        actions_raw = self.df["action"].to_numpy()
        refined_goals = goal_frame_idx.copy()
        N = self.length
        
        # Refine goals based on 'forward' action logic
        for i in range(N):
            if refined_goals[i] != -1:
                continue
            
            found = -1
            for j in range(i + 1, N):
                if actions_raw[j] == 1: # 1 is 'forward'
                    found = j
                    break
            
            if found != -1:
                refined_goals[i] = found - i
            else:
                refined_goals[i] = N - 1 - i
        
        # Calculate relative pose to refined goal
        target_indices = refined_goals + np.arange(N)
        
        T_body_goal = T_body_b[target_indices] # [N, 4, 4]
        T_b_body = homogeneous_inv(T_body_b) # [N, 4, 4]
        T_b_goal = T_b_body @ T_body_goal # [N, 4, 4]
        self.action_goals = self.get_poses(T_b_goal) # [N, 4]

    def get_poses(self, T: np.ndarray) -> np.ndarray:
        """
        Get poses in [x, y, z, yaw] format.
        Args:
            T: [N, 4, 4]
        Returns:
            poses: [N, 4]
        """
        # Yaw extraction (ZYX euler)
        R = T[:, :3, :3]
        yaw, pitch, roll = Rotation.from_matrix(R).as_euler('ZYX', degrees=True).T
        pos = T[:, :3, 3]
        return np.concatenate([pos, yaw[:, None]], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Iterable[tuple[dict, str]]:
        self._process_traj()
        has_reason = self.REASON_COL in self.df.columns
        for i in range(self.length):
            reason = str(self.df.iloc[i][self.REASON_COL]) if has_reason else ""
            if pd.isna(reason):
                reason = ""
            frame = {
                "annotation.human.action.task_description": np.array([self.task_idx], dtype=np.int32),
                "observation.state": self.poses_body[i],
                "video.ego_view": np.array(Image.open(self.images[i]).convert("RGB")),
                "action": np.concatenate([self.action_deltas[i], self.action_goals[i]]).astype(np.float32),
                "extra.cot": reason,
            }
            yield frame, self.task


class VLN_CE_Trajectories(Trajectories):
    FPS: int = 10
    ROBOT_TYPE: str = "lerobot"
    INSTRUCTION_KEY: str = "annotation.human.action.task_description"


    FEATURES = {
        # The language instruction for the task.
        "annotation.human.action.task_description": {
            "dtype": "int32", # index of task
            "shape": (1,),
            "names": None,
        },
        # The drone's pose in the first frame of the trajectory.
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": {
                "axes": ["x", "y", "z", "yaw"],
            },
        },
        # The primary video feed from the drone's ego-centric camera.
        "video.ego_view": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        },
        # The action command sent to the drone.
        # first 4 values are [dx, dy, dz, dyaw] to the next frame
        # last 4 values are goal pose of the to the current frame
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "axes": ["x", "y", "z", "yaw", "farthest_x", "farthest_y", "farthest_z", "farthest_yaw"],
            },
        },
        # Per-frame chain-of-thought reasoning from CoT annotations.
        "extra.cot": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }

    def __init__(self, data_path: str, get_task_idx: Callable[[str], int]):
        self.data_path = Path(data_path)
        self.get_task_idx = get_task_idx
        self.parquet_files = []
        
        # Find all 'data' directories.
        # Structure is assumed to be .../scene_id/data/chunk-XXX/episode.parquet
        # We verify that 'data' is a sibling of 'meta' and 'videos' to confirm it's a valid scene directory.
        for data_dir in tqdm(self.data_path.rglob("data"), desc="Scanning data directories"):
            if not data_dir.is_dir():
                continue
            
            scene_dir = data_dir.parent
            if (scene_dir / "meta").exists() and (scene_dir / "videos").exists():
                for chunk_dir in data_dir.glob("chunk-*"):
                    if chunk_dir.is_dir():
                        self.parquet_files.extend(chunk_dir.glob("*.parquet"))

    def __len__(self) -> int:
        return len(self.parquet_files)

    def __iter__(self) -> Iterable[Traj]:
        # Cache for metadata: path -> dict of episode_index -> task
        metadata_cache = {}

        for parquet_file in self.parquet_files:
            # We enforce the structure: .../scene_id/data/chunk-XXX/episode_XXXXXX.parquet
            # So traversing up 3 levels gives us the scene directory.
            chunk_dir = parquet_file.parent
            data_dir = chunk_dir.parent
            scene_dir = data_dir.parent 

            # Load metadata
            meta_file = scene_dir / "meta" / "episodes.jsonl"
            meta_key = str(meta_file)
            if meta_key not in metadata_cache:
                episodes_map = {}
                if meta_file.exists():
                    with open(meta_file, "r") as f:
                        for line in f:
                            try:
                                item = json.loads(line)
                                # task is a list of strings, we take the first one
                                if "tasks" in item and len(item["tasks"]) > 0:
                                    episodes_map[item["episode_index"]] = item["tasks"][0]
                            except Exception:
                                continue
                metadata_cache[meta_key] = episodes_map

            # Get episode index from filename (episode_000000.parquet)
            try:
                # Format is episode_XXXXXX.parquet
                episode_idx_str = parquet_file.stem.split("_")[1]
                episode_idx = int(episode_idx_str)
            except (IndexError, ValueError):
                continue

            task = metadata_cache[meta_key].get(episode_idx, "")
            task_idx = self.get_task_idx(task)

            # Get images
            # .../videos/chunk-XXX/
            chunk_name = chunk_dir.name
            video_chunk_dir = scene_dir / "videos" / chunk_name
            
            # Primary path as per instruction
            images_dir = video_chunk_dir / "observation.images.rgb.125cm_0deg"

            # Image pattern: episode_000000_0.jpg
            # The last number is the frame index
            image_prefix = f"episode_{episode_idx_str}_"
            images = sorted(
                list(images_dir.glob(f"{image_prefix}*.jpg")),
                key=lambda p: int(p.stem.split("_")[-1])
            )

            try:
                traj = VLN_CE_Traj(parquet_file, images, task=task, task_idx=task_idx)
                yield traj
            except Exception:
                print(f"Failed to load trajectory from {parquet_file}")
                import traceback
                traceback.print_exc()
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Failed to load trajectory from {parquet_file}\n")
                    log_file.write(traceback.format_exc())
                    log_file.write("\n")
                continue

    @property
    def schema(self) -> dict:
        return VLN_CE_Trajectories.FEATURES

if __name__ == "__main__":
    def get_task_idx_mock(task: str) -> int:
        return 0
    
    trajs = VLN_CE_Trajectories("/data-10T/InternData-N1/r2r", get_task_idx=get_task_idx_mock)

    i = 0
    for traj in trajs:
        i += 1
        if i >= 5:
            break
