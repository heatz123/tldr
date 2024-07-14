"""A replay buffer that efficiently stores and can sample whole paths."""

import collections

import numpy as np


class PathBufferEx:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.

    """

    def __init__(
        self,
        capacity_in_transitions,
        pixel_shape,
        max_episode_length=1000,
        use_goal=0,
    ):
        self._capacity = capacity_in_transitions
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}

        if pixel_shape is not None:
            self._pixel_dim = np.prod(pixel_shape)
        else:
            self._pixel_dim = None
        self._pixel_keys = ["obs", "next_obs"]

        self.indices_to_episode_timesteps = np.array([], dtype=np.int32)
        self.epilengths = np.array([], dtype=np.int32)

        self.max_episode_length = max_episode_length

        self.use_goal = use_goal
        if self.use_goal:
            self._pixel_keys.append("options")
            self._pixel_keys.append("next_options")

    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).

        Raises:
            ValueError: If a key is missing from path or path has wrong shape.

        """
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while self._path_segments and self._segments_overlap(
            first_seg, self._path_segments[0][0]
        ):
            self._path_segments.popleft()
        while self._path_segments and self._segments_overlap(
            second_seg, self._path_segments[0][0]
        ):
            self._path_segments.popleft()
        self._path_segments.append((first_seg, second_seg))

        # add path keys
        path["timesteps"] = np.arange(path_len, dtype=np.int64)[:, None]
        path["epilength"] = np.full([path_len], path_len, dtype=np.int64)[:, None]

        for key, array in path.items():
            if self._pixel_dim is not None and key in self._pixel_keys:
                pixel_key = f"{key}_pixel"
                state_key = f"{key}_state"
                if pixel_key not in self._buffer:
                    self._buffer[pixel_key] = np.random.randint(
                        0, 255, (self._capacity, self._pixel_dim), dtype=np.uint8
                    )  # For memory preallocation
                    self._buffer[state_key] = np.zeros(
                        (self._capacity, array.shape[1] - self._pixel_dim),
                        dtype=array.dtype,
                    )
                self._buffer[pixel_key][first_seg.start : first_seg.stop] = array[
                    : len(first_seg), : self._pixel_dim
                ]
                self._buffer[state_key][first_seg.start : first_seg.stop] = array[
                    : len(first_seg), self._pixel_dim :
                ]
                self._buffer[pixel_key][second_seg.start : second_seg.stop] = array[
                    len(first_seg) :, : self._pixel_dim
                ]
                self._buffer[state_key][second_seg.start : second_seg.stop] = array[
                    len(first_seg) :, self._pixel_dim :
                ]
            else:
                buf_arr = self._get_or_allocate_key(key, array)
                buf_arr[first_seg.start : first_seg.stop] = array[: len(first_seg)]
                buf_arr[second_seg.start : second_seg.stop] = array[len(first_seg) :]

        if second_seg.stop != 0:
            self._first_idx_of_next_path = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop

        self._transitions_stored = min(
            self._capacity, self._transitions_stored + path_len
        )

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """

        idx = np.random.choice(self._transitions_stored, batch_size)

        if self._pixel_dim is not None:
            ret_dict = {}
            keys = set(self._buffer.keys())
            for key in self._pixel_keys:
                pixel_key = f"{key}_pixel"
                state_key = f"{key}_state"
                keys.remove(pixel_key)
                keys.remove(state_key)
                if self._buffer[state_key].shape[1] != 0:
                    ret_dict[key] = np.concatenate(
                        [self._buffer[pixel_key][idx], self._buffer[state_key][idx]],
                        axis=1,
                    )
                else:
                    ret_dict[key] = self._buffer[pixel_key][idx]
            for key in keys:
                ret_dict[key] = self._buffer[key][idx]
            return ret_dict
        else:
            return {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}

    def sample_transitions_with_goals(
        self,
        batch_size,
    ):
        """Sample a batch of transitions from the buffer and return with goals.

        Args:
            batch_size (int): Number of transitions to sample.
            p_future (float): Probability of using the goal from the future states.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).
        """
        # goals: goals for traj_encoder
        # options: goals for policy

        indx = np.random.randint(self.n_transitions_stored - 1, size=batch_size)

        # sample random goals
        goal_indx = np.random.randint(self.n_transitions_stored, size=batch_size)

        if self._pixel_dim is None:
            goals = self._buffer["next_obs"][goal_indx]
        else:
            goals = self._buffer["next_obs_pixel"][goal_indx]

        # sample goals from the same trajectory
        timesteps = self._buffer["timesteps"][indx].squeeze(-1)
        epilengths = self._buffer["epilength"][indx].squeeze(-1)
        initial_indx = indx - timesteps  # index for initial observations
        final_state_indx = initial_indx + epilengths - 1  # index for final observations
        final_state_indx = np.minimum(final_state_indx, self.n_transitions_stored - 1)

        futures_indx = np.random.randint(
            indx,
            final_state_indx + 1,
            size=batch_size,
        )
        cur_exploration = self._buffer["cur_exploration"][indx].squeeze(-1)
        if self._pixel_dim is None:
            futures = self._buffer["next_obs"][futures_indx]
            options = self._buffer["options"][indx]
        else:
            futures = self._buffer["next_obs_pixel"][futures_indx]
            options = self._buffer["options_pixel"][indx]

        is_future_indx = ((np.random.rand(batch_size) < 0.8) | cur_exploration).reshape(
            -1
        )  # (batch_size,)
        trajgoals_indx = np.where(
            is_future_indx,
            futures_indx,
            -1,
        )
        success = indx == trajgoals_indx
        success_rewards = success.astype(float)[:, None]
        masks = 1.0 - success.astype(float)[:, None]
        trajgoals = np.where(
            is_future_indx[:, None],
            futures,
            options,
        )

        if self._pixel_dim is not None:
            ret_dict = {}
            keys = set(self._buffer.keys())
            for key in self._pixel_keys:
                pixel_key = f"{key}_pixel"
                state_key = f"{key}_state"
                keys.remove(pixel_key)
                keys.remove(state_key)
                if self._buffer[state_key].shape[1] != 0:
                    ret_dict[key] = np.concatenate(
                        [self._buffer[pixel_key][indx], self._buffer[state_key][indx]],
                        axis=1,
                    )
                else:
                    ret_dict[key] = self._buffer[pixel_key][indx]
            for key in keys:
                ret_dict[key] = self._buffer[key][indx]
        else:
            ret_dict = {key: buf_arr[indx] for key, buf_arr in self._buffer.items()}

        assert (
            goals.ndim
            == success_rewards.ndim
            == masks.ndim
            == ret_dict["rewards"].ndim
            == 2
        )

        ret_dict["goals"] = goals  # for traj_encoder
        ret_dict["options"] = trajgoals
        ret_dict["next_options"] = trajgoals
        ret_dict["dones"] = np.zeros_like(success_rewards)
        ret_dict["dones_exp"] = np.zeros_like(success_rewards)

        ret_dict["success_rewards"] = success_rewards
        ret_dict["masks"] = masks
        return ret_dict

    def _next_path_segments(self, n_indices):
        """Compute where the next path should be stored.

        Args:
            n_indices (int): Path length.

        Returns:
            tuple: Lists of indices where path should be stored.

        Raises:
            ValueError: If path length is greater than the size of buffer.

        """
        if n_indices > self._capacity:
            raise ValueError("Path is too long to store in buffer.")
        start = self._first_idx_of_next_path
        end = start + n_indices
        if end > self._capacity:
            second_end = end - self._capacity
            return (range(start, self._capacity), range(0, second_end))
        else:
            return (range(start, end), range(0, 0))

    def _get_or_allocate_key(self, key, array):
        """Get or allocate key in the buffer.

        Args:
            key (str): Key in buffer.
            array (numpy.ndarray): Array corresponding to key.

        Returns:
            numpy.ndarray: A NumPy array corresponding to key in the buffer.

        """
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = np.zeros((self._capacity, array.shape[1]), array.dtype)
            self._buffer[key] = buf_arr
        return buf_arr

    def clear(self):
        """Clear buffer."""
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        self._path_segments.clear()
        self._buffer.clear()

    @staticmethod
    def _get_path_length(path):
        """Get path length.

        Args:
            path (dict): Path.

        Returns:
            length: Path length.

        Raises:
            ValueError: If path is empty or has inconsistent lengths.

        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError(
                    "path has inconsistent lengths between "
                    "{!r} and {!r}.".format(length_key, key)
                )
        if not length:
            raise ValueError("Nothing in path")
        return length

    @staticmethod
    def _segments_overlap(seg_a, seg_b):
        """Compute if two segments overlap.

        Args:
            seg_a (range): List of indices of the first segment.
            seg_b (range): List of indices of the second segment.

        Returns:
            bool: True iff the input ranges overlap at at least one index.

        """
        # Empty segments never overlap.
        if not seg_a or not seg_b:
            return False
        first = seg_a
        second = seg_b
        if seg_b.start < seg_a.start:
            first, second = seg_b, seg_a
        assert first.start <= second.start
        return first.stop > second.start

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return int(self._transitions_stored)
