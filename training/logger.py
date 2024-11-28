from datetime import datetime
import json

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'{log_dir}/training_log_{self.timestamp}.json'
        self.episode_data = []
        
    def log_episode(self, episode, total_reward, steps, epsilon, loss):
        data = {
            'episode': episode,
            'total_reward': float(total_reward),
            'steps': steps,
            'epsilon': float(epsilon),
            'loss': float(loss) if loss is not None else None
        }
        self.episode_data.append(data)
        self.save_logs()
        
    def save_logs(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.episode_data, f, indent=2)