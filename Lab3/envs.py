from collections import deque

import gymnasium as gym
import numpy as np
from PIL import Image


class RecordEpisode(gym.Wrapper):
    """Registra il primo episodio con imageio/ffmpeg, frame per frame."""
    def __init__(self, env, folder):
        super().__init__(env)
        self.folder = folder
        self.writer = None
        self.recorded = False

    def reset(self, *, seed=None, options=None):
        self.finish()
        observation, info = self.env.reset(seed=seed, options=options)
        if not self.recorded:
            import imageio.v2 as imageio
            self.folder.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(str(self.folder / "episode_0.mp4"),
                                             fps=self.metadata.get("render_fps", 30), macro_block_size=1)
            self.recorded = True
            self.writer.append_data(self.env.render())
        return observation, info

    def step(self, action):
        transition = self.env.step(action)
        if self.writer is not None:
            self.writer.append_data(self.env.render())
            if transition[2] or transition[3]:
                self.finish()
        return transition

    def finish(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self):
        self.finish()
        self.env.close()


class RacingFrames(gym.Wrapper):
    """Frame skipping e stack CHW uint8, senza perdere i flag dell'ambiente."""
    def __init__(self, env, frames=4, skip=4):
        super().__init__(env)
        self.frames, self.skip = frames, skip
        self.history = deque(maxlen=frames)
        self.observation_space = gym.spaces.Box(0, 255, shape=(frames, 84, 84), dtype=np.uint8)

    @staticmethod
    def preprocess(observation):
        return np.asarray(Image.fromarray(observation[:84, 6:90]).convert("L").resize((84, 84)), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        frame = self.preprocess(observation)
        self.history.clear()
        self.history.extend(frame.copy() for _ in range(self.frames))
        return np.stack(self.history), info

    def step(self, action):
        total = 0.0
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(int(action))
            total += reward
            if terminated or truncated:
                break
        self.history.append(self.preprocess(observation))
        return np.stack(self.history), total, terminated, truncated, info


def make_env(name, video_folder=None, frames=4, skip=4):
    kwargs = {"continuous": False} if name == "CarRacing-v3" else {}
    if video_folder:
        kwargs["render_mode"] = "rgb_array"
    env = gym.make(name, **kwargs)
    if video_folder:
        # Registra prima del wrapper: mantiene risoluzione e framerate nativi.
        env = RecordEpisode(env, video_folder)
    return RacingFrames(env, frames, skip) if name == "CarRacing-v3" else env
