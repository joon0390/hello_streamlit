import os, json
import numpy as np
from training.logger import TrainingLogger
from utils.replay_buffer import ReplayBuffer
from config import *
from datetime import datetime
from colorama import init, Fore, Style

def setup_directories():
    """필요한 디렉토리 생성"""
    dirs = ['weights', 'logs']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")

def save_training_stats(episode, reward, loss, time_stamp):
    """학습 통계 저장"""
    stats = {
        'episode': episode,
        'reward': reward,
        'loss': loss,
        'timestamp': time_stamp
    }
    
    log_file = f'logs/training_stats_{datetime.now().strftime("%Y%m%d")}.json'
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(stats)
    
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=4)

def train_pointer_dqn(env, agent, config=None):
    if config is None:
        config = CONFIG
        
    # 디렉토리 설정
    setup_directories()
    
    print(f"{Fore.CYAN}Starting training with PointerDQN...{Style.RESET_ALL}")
    
    # Initialize trackers
    logger = TrainingLogger()
    episode_rewards = []
    episode_steps = []
    total_reward = 0
    episode = 0
    best_reward = float('-inf')
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config['replay_buffer_size'])
    epsilon = config['epsilon_start']
    
    try:
        for episode in range(config['num_episodes']):
            # Initialize episode
            start_pos = env.get_random_start_point()
            state = env.reset(start_pos)
            total_reward = 0
            losses = []
            steps = 0
            found_road = False
            
            while steps < config['max_steps']:
                # Action selection using PointerDQN
                action = agent.get_action(state, epsilon)
                
                # Environment interaction
                next_state, reward, done = env.step(action)
                current_pos = env.current_position
                
                # Store experience
                replay_buffer.push(state, action, reward, next_state, done)
                
                # Training
                if len(replay_buffer) > config['min_replay_size']:
                    batch = replay_buffer.sample(config['batch_size'])
                    loss = agent.train_step(batch)
                    losses.append(loss)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Post-episode updates
            if episode % config['target_update_freq'] == 0:
                agent.update_target_network()
            
            epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
            
            # Record episode results
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            avg_loss = np.mean(losses) if losses else 0
            logger.log_episode(episode, total_reward, steps, epsilon, avg_loss)
            
            # Save training stats
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_training_stats(episode, total_reward, avg_loss, time_stamp)
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save('weights/best_model_pointer.pth')
                print(f"\n{Fore.GREEN}New best model saved! Reward: {best_reward:.2f}{Style.RESET_ALL}")
            
            # Print progress and save periodically
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{config['num_episodes']}, "
                      f"Reward: {total_reward:.2f}, "
                      f"Steps: {steps}, "
                      f"Epsilon: {epsilon:.3f}, "
                      f"Loss: {avg_loss:.3f}")
                
                # 중간 모델 저장
                agent.save(f'weights/model_checkpoint_ep{episode+1}.pth')
                
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Training interrupted...{Style.RESET_ALL}")
        
    finally:
        # 최종 모델 저장
        agent.save('weights/final_model_pointer.pth')
        print(f"\n{Fore.GREEN}Training completed{Style.RESET_ALL}")
        print(f"Best reward: {best_reward:.2f}")
        
        # 최종 통계 저장
        final_stats = {
            'total_episodes': episode + 1,
            'best_reward': best_reward,
            'final_epsilon': epsilon,
            'average_reward': np.mean(episode_rewards),
            'average_steps': np.mean(episode_steps)
        }
        
        with open('logs/training_final_stats.json', 'w') as f:
            json.dump(final_stats, f, indent=4)
            
    return agent