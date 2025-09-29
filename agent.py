# agent.py
import numpy as np


class ModerationAgent:
    """
    The agent makes decisions based on a unified feature vector from any content type
    and learns using a Q-table.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_rate=0.995):
        # Expanded actions for more granular flagging
        self.actions = ['accept', 'flag_spam', 'flag_nsfw', 'flag_plagiarism', 'flag_irrelevant']
        self.q_table = {}

        # RL Hyperparameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate

        self.last_state = None
        self.last_action = None

    def _get_state(self, feature_vector: np.ndarray) -> tuple:
        """Converts the numerical feature vector into a hashable state for the Q-table."""
        # Discretize the continuous vector by rounding to create a manageable number of states
        return tuple(np.round(feature_vector, 1))

    def choose_action(self, feature_vector: np.ndarray) -> str:
        """Chooses an action based on the unified feature vector."""
        self.last_state = self._get_state(feature_vector)

        if np.random.rand() < self.epsilon:
            self.last_action = np.random.choice(self.actions)
        else:
            q_values = [self.q_table.get((self.last_state, a), 0) for a in self.actions]
            self.last_action = self.actions[np.argmax(q_values)]

        self.epsilon *= self.epsilon_decay
        return self.last_action

    def learn(self, reward: float):
        """Updates the Q-table based on the reward."""
        if self.last_state is None or self.last_action is None:
            return
        old_value = self.q_table.get((self.last_state, self.last_action), 0)
        self.q_table[(self.last_state, self.last_action)] = old_value + self.lr * (reward - old_value)
        self.last_state, self.last_action = None, None

    def get_quality_score(self, content_data: dict) -> int:
        """
        Calculates a more nuanced score based on different content criteria.
        """
        score = 100
        text = content_data.get('text')

        if text:
            # Score clarity and format
            if len(text) < 25: score -= 30
            if len(text.split()) < 5: score -= 20
        else:
            # Penalize posts with no text content
            score -= 40

        # Score relevance (simulated)
        if text and content_data.get('topic') and content_data.get('topic') not in text:
            score -= 50  # Penalize if text is irrelevant to the stated topic

        # Ensure score is within bounds
        return max(0, min(100, int(score)))